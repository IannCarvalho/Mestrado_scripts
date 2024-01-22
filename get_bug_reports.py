import json
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from bugzilla import RHBugzilla3
from util import initialize_mongo, log, verify_log
from concurrent.futures import ProcessPoolExecutor

# Definindo constantes para URLs e caminho do arquivo de log
URL_GERRIT = 'https://git.eclipse.org/r'
BUGZILLA_URL = 'https://bugs.eclipse.org/bugs'
LOG_FILE = '../logs/log_requests.txt'
NUM_CHUNKS = 24

# Conectando ao banco de dados
mongo = initialize_mongo("bug_reports")

# Inicializando a conexão com o Bugzilla
bzapi = RHBugzilla3(f"{BUGZILLA_URL}/xmlrpc.cgi")

def get_all_ids(start_year, end_year):
    now = datetime.now()
    actual_year = now.year

    all_ids = []
    log(f'Buscando IDs de bug reports até {now}', LOG_FILE)
    for year in range(start_year, end_year + 1):
        if year == actual_year:
            chfieldto = 'Now'
        else:
            chfieldto = f'{year}-12-31'

        # Fazendo uma solicitação GET para obter os IDs dos relatórios de bugs do Bugzilla
        url = f"{BUGZILLA_URL}/buglist.cgi?bug_status=CLOSED&chfieldfrom={year}-01-01&chfieldto={chfieldto}&limit=0&resolution=FIXED"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extraindo os IDs dos relatórios de bugs do HTML da resposta
        year_ids = [int(element.text.strip()) for element in soup.find_all('td', class_='first-child bz_id_column')]
        all_ids.extend(year_ids)
        log(f'Verificados {len(year_ids)} bug reports do ano de {year}', LOG_FILE)

    log(f'Foram recuperados {len(all_ids)} bug reports de {start_year} até {end_year}', LOG_FILE)
    return chunk_data(all_ids, NUM_CHUNKS)

def get_bug_reports(all_ids):
    bug_reports = []
    for bugzilla_id in all_ids:
        # Executando a consulta para obter os dados do relatório de bugs
        try:
            bug = bzapi.getbug(bugzilla_id)
            comments = bug.getcomments()
            bug = vars(bug)

            keys_to_delete = ['bugzilla', '_rawdata', 'autorefresh', '_aliases']
            for key in keys_to_delete:
                bug.pop(key, None)

            keys_to_cast = ['last_change_time', 'creation_time']
            for key in keys_to_cast:
                bug[key] = datetime.strptime(bug[key], "%Y%m%dT%H:%M:%S").isoformat() + "Z"

            keys_to_cast = ['time', 'creation_time']
            for comment in comments:
                for key in keys_to_cast:
                    comment[key] = datetime.strptime(comment[key], "%Y%m%dT%H:%M:%S").isoformat() + "Z"
        except:
            url = f"{BUGZILLA_URL}/rest/bug?id={bugzilla_id}" 
            bug = requests.get(url).json()['bugs'][0]
            url = f"{BUGZILLA_URL}/rest/bug/{bugzilla_id}/comment"
            comments = requests.get(url).json()['bugs'][str(bugzilla_id)]['comments']

        bug['comments'] = comments
        bug_report = get_gerrit_data(bug)
        
        if bug_report:
            bug_reports.append(bug_report)
            
    mongo.insert_many(bug_reports)
    return bug_reports

# Função para fazer uma solicitação HTTP e analisar a resposta JSON
def     make_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content[5:-1]
        result = json.loads(content)
        return result
    except Exception as e:
        log(f"Ocorreu um problema ao fazer a solicitação HTTP: {e}", LOG_FILE)
        return None

# Função para obter dados do Gerrit e adicioná-los ao MongoDB
def get_gerrit_data(bug_report):
    id_bugzilla = bug_report['id']
    see_also = bug_report.get("see_also", [])
    changes = {}

    for info in see_also:
        if info.startswith(URL_GERRIT):
            info = info.rstrip('/')  # Remove a barra final, se houver
            id_gerrit = info.split("/")[-1]

            try:
                project_name = make_request(f"{URL_GERRIT}/changes/{id_gerrit}/detail")['project']
                files = make_request(f"{URL_GERRIT}/changes/{id_gerrit}/revisions/current/files")
                files = list(files.keys())
                files.remove("/COMMIT_MSG")

                if project_name in changes:
                    changes[project_name] += files
                else:
                    changes[project_name] = files
            except:            
                log(f"Ocorreu um problema de validação com o Gerrit com o bug report {id_bugzilla}", LOG_FILE)
                changes = {} 
                break

    # Adicionando o relatório de bugs ao MongoDB se houver alterações do Gerrit associadas a ele
    if changes:
        log(f"Realizada a recuperação do bug report {id_bugzilla}", LOG_FILE)
        bug_report['changes'] = changes
        return bug_report
    else:
        log(f"Ignorando bug report {id_bugzilla} por não ter conexão com o Gerrit", LOG_FILE)
        

# Função principal que obtém todos os IDs dos relatórios de bugs e recupera os dados associados a cada ID
def chunk_data(data, num_chunks):
    chunk_size = len(data) // num_chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks

def main():
    verify_log(LOG_FILE)
    chunks = get_all_ids(2014, 2023)

    with ProcessPoolExecutor() as executor:
        for chunk in chunks:
            executor.submit(get_bug_reports, chunk)

if __name__ == "__main__":
    main()
