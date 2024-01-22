from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from util import read_pickle, log, verify_log, initialize_mongo, get_bug_reports

LOG_FILE = '../logs/log_archs.txt'
BATCH_SIZE = 10000
NUM_PROCESSES = 20

PRIORITIES = {'P5': 0, 'P4': 1, 'P3': 2, 'P2': 3, 'P1': 4}
SEVERITIES = {'blocker': 6, 'critical': 5, 'major': 4, 'normal': 3, 'minor': 2, 'trivial': 1, 'enhancement': 0}

TF_IDF = read_pickle('../vectors/tf-idf.pkl')
T5 = read_pickle('../vectors/t5.pkl')
BUG_REPORTS = {br['id']: br for br in get_bug_reports()}

verify_log(LOG_FILE)

def calculate_percentage_in_common(common, total):
    return common / total if total > 0 else 0

def mean_percentage(br1_percentage, br2_percentage):
    return (br1_percentage + br2_percentage) / 2

def get_number_of_files(br):
    return sum(len(files) for files in br.values())

def calculate_similarity_tf_idf(br1, br2):
    similarities = {}
    for text in ['summary', 'description']:
        for subkey in ['tokens', 'without_stopwords', 'lemmatized']:
            similarities[f'similarity_tf_idf_{text}_{subkey}'] = float(cosine_similarity(br1[text][subkey], br2[text][subkey])[0][0])
    return similarities

def calculate_similarity_repositories(br1, br2):
    br1_keys, br2_keys = set(br1.keys()), set(br2.keys())

    repositories_intersection = br1_keys & br2_keys
    repositories_in_common = len(repositories_intersection)

    br1_percentage_repositories = calculate_percentage_in_common(repositories_in_common, len(br1_keys))
    br2_percentage_repositories = calculate_percentage_in_common(repositories_in_common, len(br2_keys))

    br1_number_of_files, br2_number_of_files = get_number_of_files(br1), get_number_of_files(br2)

    files_in_common = sum(len(set(br1[repo]) & set(br2[repo])) for repo in repositories_intersection)

    br1_percentage_files = calculate_percentage_in_common(files_in_common, br1_number_of_files)
    br2_percentage_files = calculate_percentage_in_common(files_in_common, br2_number_of_files)

    return {
        'similarity_repositories': mean_percentage(br1_percentage_repositories, br2_percentage_repositories),
        'similarity_files': mean_percentage(br1_percentage_files, br2_percentage_files)
    }

def process_similarity_batch(ids_to_mongo, similarities, mongo):
    if BATCH_SIZE < len(similarities):
        mongo.insert_many(similarities)
        log(f'Todas as similaridades de {ids_to_mongo} foram calculadas e salvas no MongoDB', LOG_FILE)
        return [], []

    return similarities, ids_to_mongo

def calculate_and_save_similarity_data(process_id):
    mongo = initialize_mongo("archs")

    try:
        similarities = []
        ids_to_mongo = []

        for i in range(process_id, len(T5) - 1, NUM_PROCESSES):
            print(i)
            id1 = T5[i]['id']
            br1 = BUG_REPORTS[id1]
            for j in range(i + 1, len(T5)):
                id2 = T5[j]['id']
                br2 = BUG_REPORTS[id2]

                similarity_data = {
                    'bug_report_1': id1,
                    'bug_report_2': id2,
                    'similarity_product': int(br1['product'] == br2['product']),
                    'similarity_component': int(br1['component'] == br2['component']),
                    'similarity_op_sys': int(br1['op_sys'] == br2['op_sys']),
                    'distance_severity': abs(SEVERITIES[br1['severity']] - SEVERITIES[br2['severity']]),
                    'distance_priority': abs(PRIORITIES[br1['priority']] - PRIORITIES[br2['priority']]),
                    'similarity_t5_summary': float(cosine_similarity([T5[i]['summary']], [T5[j]['summary']])[0][0]),
                    'similarity_t5_description': float(cosine_similarity([T5[i]['description']], [T5[j]['description']])[0][0]),
                    **calculate_similarity_tf_idf(TF_IDF[id1], TF_IDF[id2]),
                    **calculate_similarity_repositories(br1['changes'], br2['changes'])
                }
                
                similarities.append(similarity_data)

            ids_to_mongo.append(br1['id'])
            similarities, ids_to_mongo = process_similarity_batch(ids_to_mongo, similarities, mongo)

        similarities, ids_to_mongo = process_similarity_batch(ids_to_mongo, similarities, mongo)

    except Exception as e:
        log(f'Exceção no processo {process_id}: {e}', LOG_FILE)

def main():
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        for process_id in range(NUM_PROCESSES):
            executor.submit(calculate_and_save_similarity_data, process_id)

if __name__ == "__main__":
    main()