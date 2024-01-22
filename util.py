import pickle
import pymongo
import os

MONGODB_URI = "mongodb://localhost:27017"
OUTPUT_FILE_BR = '../data/brs.pkl'

def read_pickle(path):
    with open(f"{path}.pkl", 'rb') as pickle_file:
        return pickle.load(pickle_file)
    
def save_to_pickle(data, output_file):
    with open(f"{output_file}.pkl", 'wb') as file:
        pickle.dump(data, file)
        
def get_bug_reports():
    if os.path.exists(OUTPUT_FILE_BR):
        return read_pickle(OUTPUT_FILE_BR)
    else:
        col = initialize_mongo("bug_reports")
        data = list(col.find())
        
        save_to_pickle(data, OUTPUT_FILE_BR)
        
        return list(col.find())

def count_files_in_directory(path):
    try:
        file_list = os.listdir(path)
        files = [file for file in file_list if os.path.isfile(os.path.join(path, file))]
        number_of_files = len(files)

        return number_of_files

    except OSError as e:
        print(f"Error accessing the directory: {e}")
        return None
    
def start_process(path):
    return 0, count_files_in_directory(path)

# Registra mensagens de log em um arquivo e imprime no console
def log(msg, log_path):
    with open(log_path, 'a', encoding="utf-8") as log_file:
        log_file.write(msg + '\n')
    print(msg)
    
# Verifica se o arquivo de log existe e o exclui se necessário
def verify_log(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
        
def initialize_mongo(column):
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["bugs"]
    return db[column]

def to_binary_label(label):
    return 0 if label < 0.5 else 1

def get_batch(path, batchs_processed):
    batch = read_pickle(f"{path}/batch_{batchs_processed}")
    return batch

def verify_early_stopping(val_f1, best_f1, tol, epoch, model, model_file):
    stop = False

    if val_f1 > best_f1 + tol:
        best_f1 = val_f1
        epoch += 1
        print(f'Going to epoch {epoch}')
        save_model(model, model_file)
    elif val_f1 > best_f1:
        best_f1 = val_f1
        save_model(model, model_file)
    else:
        stop = True
        print(f'Early stopping at epoch {epoch}')
        
    return stop, epoch, best_f1

def save_model(model, model_file="model"):
    model.save(f'../models/{model_file}.h5')
    print("Model saved sucessfully!")