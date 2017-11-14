import pymysql
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation,BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras
import numpy as np
import pickle
import random
import model_helpers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import *

connection = pymysql.connect(host='112.137.142.8',
                             user='root',
                             password='dhqghn',
                             db='gmail',
                             port=5306,
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor)

model_save_path = "model-data/" + LANG + "_spam_model.h5"

def build_model():
    model = Sequential()
    model.add(Dense(128,activation='relu', input_shape=(NUMBER_OF_TERMS,)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    optimizer = RMSprop(lr=1e-4)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
    print('compile done')
    return model

def fit_model(model,x,y,epochs=4):
    checkpointer = ModelCheckpoint(filepath=model_save_path,
                                    verbose=1,
                                    save_best_only=True)    

    history = model.fit(x,y,batch_size=64,epochs=epochs,verbose=1,shuffle=True, validation_split=0.01,
              callbacks=[checkpointer]).history
              
    return history


def train():
    train_docs, test_docs = model_helpers.get_data_by_language(connection, LANG, NUMBER_OF_TRAIN_DOCS,
                                                                NUMBER_OF_TEST_DOCS)

    start = time.time()

    print("Processing data...")

    model_helpers.train_tf_idf(train_docs + test_docs, NUMBER_OF_TERMS, LANG)
    features_matrix, labels = model_helpers.extract_features_by_tf_idf(train_docs, LANG)
    test_features_matrix, test_labels = model_helpers.extract_features_by_tf_idf(test_docs, LANG)

    model = build_model()

    # train model
    fit_model(model,features_matrix, labels)
    print("end {}".format(time.time()-start))
    predicted_labels = model.predict_classes(test_features_matrix).reshape(-1)

    print("Acc : {} ".format(accuracy_score(test_labels, predicted_labels)))
    print("Precision: {}".format(precision_score(test_labels, predicted_labels)))
    print("Recall {}".format(recall_score(test_labels, predicted_labels)))
    print("F1 {}".format(f1_score(test_labels, predicted_labels)))
    connection.close()

def predict(doc):
    model = build_model()
    model.load_weights("model-data/"+ doc['lang'] + "_spam_model.h5")
    feature_matrix, _ = model_helpers.extract_features_by_tf_idf([doc], doc['lang'])

    label = model.predict_classes(feature_matrix).tolist()[0]
    return label

if __name__ == "__main__":
    train()
