from sentence_transformers import SentenceTransformer
from util import save_to_pickle, log, get_bug_reports, verify_log

# Definindo constantes e caminhos de arquivo
LOG_FILE = '../logs/log_t5.txt'
OUTPUT_FILE = '../vectors/t5.pkl'
MODEL_NAME = 'sentence-transformers/sentence-t5-xxl'

# Inicializa o modelo SentenceTransformer uma vez fora dos processos
model = SentenceTransformer(MODEL_NAME)

# Vetoriza um relatório de bug, incluindo o resumo e os comentários
def vectorize_bug_report(bug_report):
    summary, description = model.encode([bug_report['summary'], bug_report['comments'][0]['raw_text']])
    
    vectors = {
        'id': bug_report['id'],
        'summary': summary,
        'description': description
    }

    log(f'Bug {bug_report["id"]} vetorizado', LOG_FILE)
    return vectors

# Função principal do programa
def main():
    verify_log(LOG_FILE)

    bug_reports = get_bug_reports()

    # Utiliza list comprehension para vetorizar os relatórios de bug
    bug_data = [vectorize_bug_report(bug_report) for bug_report in bug_reports]

    log('Escrevendo todos os vetores T5', LOG_FILE)
    save_to_pickle(bug_data, OUTPUT_FILE)  # Salva em formato binário

# Verifica se o programa está sendo executado como um script principal
if __name__ == "__main__":
    main()
