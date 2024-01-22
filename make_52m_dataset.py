from util import initialize_mongo, save_to_pickle
import pandas as pd
import math

BATCH_SIZE = 5000
MONGO = initialize_mongo("archs")
PATH = '../data/all_data'
WITHOUT_TEST = initialize_mongo("archs_without_test")

def make_df(dataset):
    df = pd.DataFrame(dataset)
    df['Group'] = pd.cut(df['similarity_files'], bins=1000, labels=False)
    print('Groups made')
    return df

def stratify_data(frac, df):
    return df.groupby('Group', group_keys=False).apply(lambda x: x.sample(frac=frac))
    
def stratify_data_with_rest(frac, df):
    data_stratified = stratify_data(frac, df)
    data_rest = df.drop(data_stratified.index)
    return data_stratified, data_rest

def get_mongo_data_and_save(data, split):
    data = data.sample(frac=1).reset_index(drop=True)
    
    if not any(data['similarity_files'] >= 0.5):
        print(f"Warning: No files with similarity_files >= 0.5 found for {split}. Exiting.")
        return
    
    similars = data[data['similarity_files'] >= 0.5].reset_index(drop=True)
    not_similars = data[data['similarity_files'] < 0.5].reset_index(drop=True)

    print(len(similars))
    num_of_batches = math.ceil(len(data) / BATCH_SIZE)
    similar_per_batch = len(similars) / num_of_batches
    print(similar_per_batch)
    add_float, num_added = 0, 0
    appended = 0
    for batch_num in range(num_of_batches):
        add_float += similar_per_batch
        num_of_add = int(add_float - num_added)
        
        qnt_not_similars = BATCH_SIZE - num_of_add

        print(similars.head(5))
        current_batch = pd.concat([similars.head(num_of_add), not_similars.head(qnt_not_similars)], ignore_index=True)        
        if len(current_batch) < BATCH_SIZE:
            current_batch = pd.concat([similars, not_similars], ignore_index=True)
            print("Last Batch")
        else:
            not_similars = not_similars.iloc[qnt_not_similars:]
            similars = similars.iloc[num_of_add:]
        print(similars.head(5))
        
        
        if not any(current_batch['similarity_files'] >= 0.5):
            print(f"Warning: No files with similarity_files >= 0.5 found in the stratified batch {batch_num} for {split}. Exiting.")
            return

        to_save = list(MONGO.find({"_id": {"$in": list(current_batch['_id'])}}))
        if split != 'test':
            save_to_pickle(to_save, f'{PATH}/{split}/batch_{batch_num}')
            WITHOUT_TEST.insert_many(to_save)
        else:
            save_to_pickle(to_save, f'../data/{split}/batch_{batch_num}')
        
        num_added += num_of_add
        
        appended += len(current_batch)
        print(f"Batch {batch_num} of {split} created")
        print(f"Total data", len(data), "Data Appended", appended)

def request_to_mongo():
    data = {'similarity_files': [], '_id': []}
    for arch in MONGO.find({"similarity_files": {"$gte": 0}}, {"similarity_files": 1, "_id": 1}):
        data['similarity_files'].append(arch['similarity_files'])
        data['_id'].append(arch['_id'])
        
    print(f'Dataset Builded')
    return data

def main():
    data = request_to_mongo()
    
    df = make_df(data)

    train_val, test  = stratify_data_with_rest(0.8, df)
    train, val = stratify_data_with_rest(0.8, train_val)
    
    get_mongo_data_and_save(test, 'test')
    get_mongo_data_and_save(train, 'train')
    get_mongo_data_and_save(val, 'val')
    

if __name__ == "__main__":
    main()
