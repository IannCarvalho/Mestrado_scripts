from util import initialize_mongo, read_pickle, save_to_pickle
import pandas as pd
import random
import sys
import os

BATCH_SIZE = 5000
MONGO = initialize_mongo("archs_without_test")
PATH = '../data/subsample/'

def list_files(directory):
    files = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def request_to_mongo(is_balanced):
    if is_balanced:
        not_similars = {'similarity_files': [], '_id': []}
        similars = {'similarity_files': [], '_id': []}

        for arch in MONGO.find({"similarity_files": {"$gte": 0}}, {"similarity_files": 1, "_id": 1}):
            if arch["similarity_files"] < 0.5:
                not_similars['similarity_files'].append(arch['similarity_files'])
                not_similars['_id'].append(arch['_id'])
            else:
                similars['similarity_files'].append(arch['similarity_files'])
                similars['_id'].append(arch['_id'])
                
        data = (not_similars, similars)
    else:
        data = {'similarity_files': [], '_id': []}
        for arch in MONGO.find({"similarity_files": {"$gte": 0}}, {"similarity_files": 1, "_id": 1}):
            data['similarity_files'].append(arch['similarity_files'])
            data['_id'].append(arch['_id'])
        
    print(f'Dataset Builded')
    return data

def make_df(dataset):
    df = pd.DataFrame(dataset)
    df['Group'] = pd.cut(df['similarity_files'], bins=1000, labels=False)
    return df

def stratify_data(frac, df):
    return df.groupby('Group', group_keys=False).apply(lambda x: x.sample(frac=frac))
    
def stratify_data_with_rest(frac, df):
    data_stratified = stratify_data(frac, df)
    data_rest = df.drop(data_stratified.index)
    return data_stratified, data_rest

def get_mongo_data_and_save(data, split):
    data = list(data)
    random.shuffle(data)
    print(PATH)
    for batch_num, batch in enumerate(range(0, len(data), BATCH_SIZE)):
        current_batch = data[batch:batch + BATCH_SIZE]
        to_save = list(MONGO.find({"_id": {"$in": current_batch}}))
        save_to_pickle(to_save, f'{PATH}{split}/batch_{batch_num}')
        print(f"Batch {batch_num} of {split} created")

def main(is_balanced):
    global PATH
    data = request_to_mongo(is_balanced)
    
    if is_balanced:
        PATH += '/750x_balanced/'
        print(PATH)
        not_similars, similars = data
        df = make_df(not_similars)
        
        num_similars, num_not_similars = len(similars['similarity_files']), len(not_similars['similarity_files'])
        
        frac = 750 * (num_similars/num_not_similars)

        stratified_not_similars = stratify_data(frac, df)
        
        not_similars_train, not_similars_val = stratify_data_with_rest(0.8, stratified_not_similars)

        df = make_df(similars)

        similars_train, similars_val = stratify_data_with_rest(0.8, df)
        
        get_mongo_data_and_save(list(similars_train['_id']) + list(not_similars_train['_id']), 'train')
        get_mongo_data_and_save(list(similars_val['_id']) + list(not_similars_val['_id']), 'val')
    else:
        desired_number = 0
        for split in ['train', 'val']:
            for batch_num in range(len(list_files(PATH + '/balanced/' + split))):
                batch = read_pickle(f'{PATH}/balanced/{split}/batch_{batch_num}')
                desired_number += len(batch)
                     
        PATH += '/desbalanced/'
        
        df = make_df(data)

        stratified = stratify_data(desired_number/len(data['similarity_files'] ), df)

        train, val = stratify_data_with_rest(0.8, stratified)
        
        get_mongo_data_and_save(list(train['_id']), 'train')
        get_mongo_data_and_save(list(val['_id']), 'val')

if __name__ == "__main__":
    is_balanced = eval(sys.argv[-1])  # Convertendo para booleano
    main(is_balanced)