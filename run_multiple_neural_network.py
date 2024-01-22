from neural_network import SimilarityModelTrainer

SimilarityModelTrainer(all_data=True,  balanced='',             embeddings=False, preprocess=False, with_t5=False, with_tf_idf=False, model_name='1.52m simple').run()
SimilarityModelTrainer(all_data=False, balanced='desbalanced',  embeddings=False, preprocess=False, with_t5=False, with_tf_idf=False, model_name='2.10m simple').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess=False, with_t5=False, with_tf_idf=False, model_name='3.10m balanced simple').run()

SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess=False,               with_t5=False, with_tf_idf=True, model_name='4.10m balanced cosine distance tf-idf').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess='without_stopwords', with_t5=False, with_tf_idf=True, model_name='5.10m balanced cosine distance tf-idf without stopwords').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess='lemmatized',        with_t5=False, with_tf_idf=True, model_name='6.10m balanced cosine distance tf-idf lemmatized').run()

SimilarityModelTrainer(all_data=True, balanced='',     embeddings=False, preprocess='lemmatized',        with_t5=False, with_tf_idf=True, model_name='Test PR AUC total data Precision').run()

SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess=False,                with_t5=False, with_tf_idf=True, model_name='7.10m balanced embeddings tf-idf').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess='without_stopwords',  with_t5=False, with_tf_idf=True, model_name='8.10m balanced embeddings tf-idf without stopwords').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess='lemmatized',         with_t5=False, with_tf_idf=True, model_name='9.10m balanced embeddings tf-idf lemmatized').run()

SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess=False,               with_t5=True, with_tf_idf=False, model_name='10.10m balanced cosine distance t5').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess='without_stopwords', with_t5=True, with_tf_idf=False, model_name='11.10m balanced cosine distance t5 without stopwords').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess='lemmatized',        with_t5=True, with_tf_idf=False, model_name='12.10m balanced cosine distance t5 lemmatized').run()

SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess=False,                with_t5=True, with_tf_idf=False, model_name='13.10m balanced embeddings t5').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess='without_stopwords',  with_t5=True, with_tf_idf=False, model_name='14.10m balanced embeddings t5 without stopwords').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess='lemmatized',         with_t5=True, with_tf_idf=False, model_name='15.10m balanced embeddings t5 lemmatized').run()

SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess=False,               with_t5=True, with_tf_idf=True, model_name='16.10m balanced cosine distance tf-idf+t5').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess='without_stopwords', with_t5=True, with_tf_idf=True, model_name='17.10m balanced cosine distance tf-idf+t5 without stopwords').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=False, preprocess='lemmatized',        with_t5=True, with_tf_idf=True, model_name='18.10m balanced cosine distance tf-idf+t5 lemmatized').run()

SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess=False,                with_t5=True, with_tf_idf=True, model_name='19.10m balanced embeddings tf-idf+t5').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess='without_stopwords',  with_t5=True, with_tf_idf=True, model_name='20.10m balanced embeddings tf-idf+t5 without stopwords').run()
SimilarityModelTrainer(all_data=False, balanced='balanced',     embeddings=True, preprocess='lemmatized',         with_t5=True, with_tf_idf=True, model_name='21.10m balanced embeddings tf-idf+t5 lemmatized').run()