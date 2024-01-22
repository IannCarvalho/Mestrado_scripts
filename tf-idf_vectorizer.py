import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from util import save_to_pickle, log, get_bug_reports, verify_log

# Baixar os pacotes necessários do NLTK
nltk.download('punkt')  # Baixa o tokenizador punkt do NLTK
nltk.download('wordnet')  # Baixa o WordNet do NLTK
nltk.download('stopwords')  # Baixa as stopwords do NLTK

# Inicializar o lematizador e o conjunto de stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Inicializar o vetorizador TF-IDF
vectorizer = TfidfVectorizer()

# Definindo constantes e caminhos de arquivo
LOG_FILE = '../logs/log_tf-idf.txt'  # Caminho do arquivo de log
OUTPUT_FILE = '../vectors/tf-idf.pkl'  # Caminho do arquivo de saída dos vetores TF-IDF

# Removendo caracteres especiais
def remove_special_characters(text):
    special_characters = re.compile('[^a-zA-Z0-9 ]')
    return special_characters.sub('', text)

# Tokenizando texto
def tokenize(text):
    return word_tokenize(text)

# Removendo stopwords de tokens recebidos
def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

# Lematizando tokens recebidos
def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# vetorizando texto
def vectorize_data(data):
    return vectorizer.fit_transform(data)

# Realiza toddas as etapas de preprocessamento armazenando todas elas vetorizadas
def preprocess_text(texts):
    tokenized_data = []
    without_stopwords_data = []
    lemmatized_data = []
    
    for text in texts:
        text = remove_special_characters(text).lower()
        
        tokens = tokenize(text)
        without_stopwords = remove_stopwords(tokens)
        all_lemmatized = lemmatize(without_stopwords)
        
        if tokens and without_stopwords and any(not lemmatized.isnumeric() for lemmatized in all_lemmatized):
            tokenized_data.append(" ".join(tokens))
            without_stopwords_data.append(" ".join(without_stopwords))
            lemmatized_data.append(" ".join(all_lemmatized))
        else:
            tokenized_data.append("")
            without_stopwords_data.append("")
            lemmatized_data.append("")

    return [vectorize_data(tokenized_data), vectorize_data(without_stopwords_data), vectorize_data(lemmatized_data)]

def main():
    # Verifica se o arquivo de log existe e o exclui se necessário
    verify_log(LOG_FILE)  # Verifica e, se existir, exclui o arquivo de log

    bug_reports = get_bug_reports()

    summaries = []
    descriptions = []
    ids = []
    
    for bug_report in bug_reports:
        summaries.append(bug_report['summary'])
        descriptions.append(bug_report['comments'][0]['raw_text'])
        ids.append(bug_report['id'])
    
    summaries = preprocess_text(summaries)
    descriptions = preprocess_text(descriptions)

    bug_data = {}
    for i, report_id in enumerate(ids):
        bug_data[report_id] = {
            'id': report_id,
            'summary': {
                'tokens': summaries[0][i].toarray()[0],
                'without_stopwords': summaries[1][i].toarray()[0],
                'lemmatized': summaries[2][i].toarray()[0]
            },
            'description': {
                'tokens': descriptions[0][i].toarray()[0],
                'without_stopwords': descriptions[1][i].toarray()[0],
                'lemmatized': descriptions[2][i].toarray()[0]
            }
        }

    log('Escrevendo todos os vetores TF-IDF', LOG_FILE)
    save_to_pickle(bug_data, OUTPUT_FILE)  # Salva em formato binário

if __name__ == "__main__":
    main()
