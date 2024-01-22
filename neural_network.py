from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, precision_score
from util import (
    save_to_pickle,
    read_pickle,
    start_process,
    get_batch,
    to_binary_label,
    save_model
)
from util_neural_network import (
    cossine_distance_model,
    t5_tf_idf_model,
    tf_idf_model,
    t5_model
)
import numpy as np

np.random.seed(42)

class SimilarityModelTrainer:
    def __init__(self, all_data=False, balanced=True, embeddings=False, preprocess=None, with_t5=True, with_tf_idf=True, epochs=1000, n_iter_no_change=5, model_name='Model'):
        self.SCALER = StandardScaler()
        self.EPOCHS = epochs
        self.DATASET = 'all_data/' if all_data else 'subsample/'
        self.BALANCED = '' if all_data else balanced
        self.EMBEDDINGS = embeddings
        self.WITH_T5 = with_t5
        self.WITH_TF_IDF = with_tf_idf
        self.PREPROCESS = preprocess if preprocess else 'tokens'
        self.PATH = f"../data/{self.DATASET}/{self.BALANCED}"
        self.MAX_ITER_NO_CHANGE = n_iter_no_change
        self.MODEL_NAME = model_name
        self.NUMERIC_FEATURES = [
            "distance_severity",
            "distance_priority",
            "similarity_op_sys",
            "similarity_component",
            "similarity_product",
        ]
        if embeddings:
            if with_tf_idf:
                self.TF_IDF = read_pickle('../vectors/tf-idf')
            if with_t5:
                self.T5 = {br['id']: br for br in read_pickle('../vectors/t5')}
        else:
            if with_tf_idf:
                self.NUMERIC_FEATURES += [
                    "similarity_tf_idf_summary_" + self.PREPROCESS,
                    "similarity_tf_idf_description_" + self.PREPROCESS,
                ]
            if with_t5:
                self.NUMERIC_FEATURES += [
                    "similarity_t5_summary",
                    "similarity_t5_description",
                ]
        self.MODEL = self.create_model()
        
        self.best_AUC = 0
        self.stop = False
        self.n_iter_no_change = 0
        self.epoch = 1
        print(self.PATH)

    def create_model(self):    
        numeric_features_size = len(self.NUMERIC_FEATURES)
        
        def load_tf_idf_size():
            summary_size = len(self.TF_IDF[1]['summary'][self.PREPROCESS])
            description_size = len(self.TF_IDF[1]['description'][self.PREPROCESS])
            return summary_size, description_size

        if self.EMBEDDINGS:
            if self.WITH_T5 and self.WITH_TF_IDF:
                summary_size, description_size = load_tf_idf_size()
                return t5_tf_idf_model(numeric_features_size, summary_size, description_size)

            if self.WITH_TF_IDF:
                summary_size, description_size = load_tf_idf_size()
                return tf_idf_model(numeric_features_size, summary_size, description_size)

            if self.WITH_T5:
                return t5_model(numeric_features_size)

        return cossine_distance_model(numeric_features_size)

    def separate_features_labels(self, documents):
        features = []
        labels = []
        numeric_features = []
        summary_TF_IDF_1, summary_TF_IDF_2, description_TF_IDF_1, description_TF_IDF_2 = [], [], [], []
        summary_T5_1, summary_T5_2, description_T5_1, description_T5_2 = [], [], [], []

        def process_document(document):
            labels.append(to_binary_label(document['similarity_files']))
            numeric_features.append(np.array([document[feature] for feature in self.NUMERIC_FEATURES]))
            
            if self.EMBEDDINGS:
                if self.WITH_TF_IDF:
                    summary_TF_IDF_1.append(self.TF_IDF[document["bug_report_1"]]['summary'][self.PREPROCESS])
                    summary_TF_IDF_2.append(self.TF_IDF[document["bug_report_2"]]['summary'][self.PREPROCESS])
                    description_TF_IDF_1.append(self.TF_IDF[document["bug_report_1"]]['description'][self.PREPROCESS])
                    description_TF_IDF_2.append(self.TF_IDF[document["bug_report_2"]]['description'][self.PREPROCESS])
                
                if self.WITH_T5:
                    summary_T5_1.append(self.T5[document["bug_report_1"]]['summary'])
                    summary_T5_2.append(self.T5[document["bug_report_2"]]['summary'])
                    description_T5_1.append(self.T5[document["bug_report_1"]]['description'])
                    description_T5_2.append(self.T5[document["bug_report_2"]]['description'])
                    
        for document in documents:
            process_document(document)

        if self.EMBEDDINGS:
            if self.WITH_TF_IDF:
                features += [np.array(summary_TF_IDF_1), np.array(summary_TF_IDF_2), np.array(description_TF_IDF_1), np.array(description_TF_IDF_2)]
            if self.WITH_T5:
                features += [np.array(summary_T5_1), np.array(summary_T5_2), np.array(description_T5_1), np.array(description_T5_2)]
                
        numeric_features = self.SCALER.fit_transform(np.vstack(numeric_features))
        features += [np.array(numeric_features)]

        return features, np.array(labels)

    def train(self):
        path = self.PATH + "train"
        batchs_processed, total_batchs = start_process(path)
        print(path, total_batchs)
        while(batchs_processed < total_batchs):
            features, labels = self.separate_features_labels(get_batch(path, batchs_processed))
            self.MODEL.train_on_batch(x=features, y=labels)
            batchs_processed += 1
            print(f'Training: {batchs_processed/total_batchs:.2%} processed in epoch {self.epoch}.')

    def validate(self):
        path = self.PATH + "val"
        batchs_processed, total_batchs = start_process(path)
        val_AUC, val_acc, val_pre = 0, 0, 0
        
        while batchs_processed < total_batchs:
            features, labels = self.separate_features_labels(get_batch(path, batchs_processed))
            y_pred = self.MODEL.predict_on_batch(features)
            batchs_processed += 1
            
            precision_curve, recall_curve, _ = precision_recall_curve(labels, y_pred)
            val_AUC += auc(recall_curve, precision_curve)
            
            y_pred_binary = [to_binary_label(pred) for pred in y_pred]
            val_acc += accuracy_score(labels, y_pred_binary)
            val_pre += precision_score(labels, y_pred_binary)
            
            print(f'Validation: {batchs_processed/total_batchs:.2%} processed in epoch {self.epoch}.')

        val_AUC /= batchs_processed
        val_acc /= batchs_processed
        val_pre /= batchs_processed
        print(f'Val Accuracy Score: {val_acc:.5f}, Val Precision Score: {val_pre:.5f}, Val AUC Score: {val_AUC:.5f}')
        
        self.verify_early_stopping(val_AUC)
   
    def verify_early_stopping(self, val_AUC):
        self.n_iter_no_change += 1
        if val_AUC > self.best_AUC:
            self.best_AUC = val_AUC
            self.n_iter_no_change = 0
            save_model(self.MODEL, f'../models/{self.MODEL_NAME}')
        if self.n_iter_no_change >= self.MAX_ITER_NO_CHANGE:
            self.stop = True

    def test(self):
        path = "../data/test"
        batchs_processed, total_batchs = start_process(path)
        y_pred = np.array([])
        y_true = np.array([])
        while batchs_processed < total_batchs:
            features, labels = self.separate_features_labels(get_batch(path, batchs_processed))
            y_pred = np.concatenate((y_pred, np.ravel(self.MODEL.predict_on_batch(features))))
            y_true = np.concatenate((y_true, labels))
            batchs_processed += 1
            
            print(f'Test: {batchs_processed/total_batchs:.2%} processed.')
        
        save_to_pickle({"y_pred":y_pred, "y_true":y_true, "epochs":self.epoch, "best_f1":self.best_AUC}, f'../predictions/{self.MODEL_NAME}')
        print(f"{self.MODEL_NAME} saved sucessfully!")

    def run(self):
        while(not self.stop and self.epoch < self.EPOCHS):
            self.train()
            self.validate()
            if not self.stop:
                self.epoch += 1
                print(f'Going to epoch {self.epoch}')
            else:
                print(f'Early stopping at epoch {self.epoch}')
                
        self.test()