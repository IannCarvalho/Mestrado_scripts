import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Precision

tf.random.set_random_seed(42)

def cossine_distance_model(numeric_features_size):
    # Definindo a entrada
    numeric_features = Input(shape=(numeric_features_size,))

    # Camadas densas finais
    a = Dense(32, activation="relu")(numeric_features)
    a = Dense(16, activation="relu")(a)
    a = Dense(1, activation="sigmoid")(a)

    # Criando o modelo
    model = Model(inputs=numeric_features, outputs=a)

    print("To usando AUC")
    # Compilando o modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision()])

    return model

def tf_idf_model(numeric_features_size, summary_size, description_size):
    summary_TF_IDF_1        = Input(shape=(summary_size,))
    summary_TF_IDF_2        = Input(shape=(summary_size,))
    description_TF_IDF_1    = Input(shape=(description_size,))
    description_TF_IDF_2    = Input(shape=(description_size,))

    numeric_features      = Input(shape=(numeric_features_size,))

    inputs =    [summary_TF_IDF_1,
                summary_TF_IDF_2,
                description_TF_IDF_1,
                description_TF_IDF_2,
                numeric_features]

    # Subestrutura para os embeddings de 9246 dimensões
    a = Concatenate()([summary_TF_IDF_1, summary_TF_IDF_2])
    a = Dense(32, activation="relu")(a)
    a = Dense(16, activation="relu")(a)
    a = Dense(1, activation="linear")(a)

    # Subestrutura para os embeddings de 57747 dimensões
    b = Concatenate()([description_TF_IDF_1, description_TF_IDF_2])
    b = Dense(32, activation="relu")(b)
    b = Dense(16, activation="relu")(b)
    b = Dense(1, activation="linear")(b)

    # Concatenando as saídas das duas subestruturas e as 5 variáveis numéricas
    combined = Concatenate()([a, b, numeric_features])

    # Camadas densas finais
    c = Dense(32, activation="relu")(combined)
    c = Dense(16, activation="relu")(c)
    c = Dense(1, activation="sigmoid")(c)

    # Criando o modelo
    model = Model(inputs=inputs, outputs=c)

    # Compilando o modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score()])
    
    return model

def t5_model(numeric_features_size):
    summary_T5_1            = Input(shape=(768,))
    summary_T5_2            = Input(shape=(768,))
    description_T5_1        = Input(shape=(768,))
    description_T5_2        = Input(shape=(768,))

    numeric_features      = Input(shape=(numeric_features_size,))

    inputs =    [summary_T5_1,
                summary_T5_2,
                description_T5_1,
                description_T5_2,
                numeric_features]
    # Subestrutura para os embeddings de 9246 dimensões
    a = Concatenate()([summary_T5_1, summary_T5_2])
    a = Dense(32, activation="relu")(a)
    a = Dense(16, activation="relu")(a)
    a = Dense(1, activation="linear")(a)

    # Subestrutura para os embeddings de 57747 dimensões
    b = Concatenate()([description_T5_1, description_T5_2])
    b = Dense(32, activation="relu")(b)
    b = Dense(16, activation="relu")(b)
    b = Dense(1, activation="linear")(b)

    # Concatenando as saídas das duas subestruturas e as 5 variáveis numéricas
    combined = Concatenate()([a, b, numeric_features])

    # Camadas densas finais
    c = Dense(32, activation="relu")(combined)
    c = Dense(16, activation="relu")(c)
    c = Dense(1, activation="sigmoid")(c)

    # Criando o modelo
    model = Model(inputs=inputs, outputs=c)

    # Compilando o modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score()])
    
    return model

def t5_tf_idf_model(numeric_features_size, summary_size, description_size):
    summary_TF_IDF_1        = Input(shape=(summary_size,))
    summary_TF_IDF_2        = Input(shape=(summary_size,))
    description_TF_IDF_1    = Input(shape=(description_size,))
    description_TF_IDF_2    = Input(shape=(description_size,))

    summary_T5_1            = Input(shape=(768,))
    summary_T5_2            = Input(shape=(768,))
    description_T5_1        = Input(shape=(768,))
    description_T5_2        = Input(shape=(768,))

    numeric_features      = Input(shape=(numeric_features_size,))

    inputs =    [summary_TF_IDF_1,
                summary_TF_IDF_2,
                description_TF_IDF_1,
                description_TF_IDF_2,
                summary_T5_1,
                summary_T5_2,
                description_T5_1,
                description_T5_2,
                numeric_features]

    # Subestrutura para os embeddings de 9246 dimensões
    a = Concatenate()([summary_TF_IDF_1, summary_TF_IDF_2])
    a = Dense(32, activation="relu")(a)
    a = Dense(16, activation="relu")(a)
    a = Dense(1, activation="linear")(a)

    # Subestrutura para os embeddings de 57747 dimensões
    b = Concatenate()([description_TF_IDF_1, description_TF_IDF_2])
    b = Dense(32, activation="relu")(b)
    b = Dense(16, activation="relu")(b)
    b = Dense(1, activation="linear")(b)

    # Subestrutura para os embeddings de 768 dimensões
    c = Concatenate()([summary_T5_1, summary_T5_2])
    c = Dense(32, activation="relu")(c)
    c = Dense(16, activation="relu")(c)
    c = Dense(1, activation="linear")(c)

    # Subestrutura para os embeddings de 768 dimensões
    d = Concatenate()([description_T5_1, description_T5_2])
    d = Dense(32, activation="relu")(d)
    d = Dense(16, activation="relu")(d)
    d = Dense(1, activation="linear")(d)

    # Concatenando as saídas das duas subestruturas e as 5 variáveis numéricas
    combined = Concatenate()([a, b, c, d, numeric_features])

    # Camadas densas finais
    e = Dense(32, activation="relu")(combined)
    e = Dense(16, activation="relu")(e)
    e = Dense(1, activation="sigmoid")(e)

    # Criando o modelo
    model = Model(inputs=inputs, outputs=e)

    # Compilando o modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[F1Score()])
    
    return model
            
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred >= 0.5, tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_positives = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_negatives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_positives, self.dtype)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(false_positives, self.dtype)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(false_negatives, self.dtype)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-5)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-5)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
        return f1

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
    