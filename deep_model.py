import pymysql
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras
import numpy as np
import pickle
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import model_helpers

connection = pymysql.connect(host='112.137.142.8',
                             user='root',
                             password='dhqghn',
                             db='gmail',
                             port=5306,
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor)

NUMBER_OF_TRAIN_DOCS = 1000
NUMBER_OF_TEST_DOCS = 300
NUMBER_OF_TERMS = 1500
LANG = 'vi'
model_save_path = "spam_model.h5"


def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(NUMBER_OF_TERMS,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    optimizer = Adam(lr=1e-4)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    print('compile done')
    return model


def check_model(model, x, y, epochs=4):
    checkpointer = ModelCheckpoint(filepath=model_save_path,
                                   verbose=1,
                                   save_best_only=True)

    history = model.fit(x, y, batch_size=32, epochs=epochs, verbose=1, shuffle=True, validation_split=0.25,
                        callbacks=[checkpointer]).history

    return history


def main():
    train_docs, test_docs = model_helpers.get_data(connection, LANG, NUMBER_OF_TRAIN_DOCS, NUMBER_OF_TEST_DOCS)
    X_train, Y_train, X_test, Y_test = model_helpers.extract_data(train_docs, test_docs, NUMBER_OF_TERMS)

    model = SVC()

    # train model
    model.fit(X_train, Y_train)
    # model = build_model()
    # his = check_model(model,X_train,Y_train,50)
    Y_pred = model.predict(X_test)
    # Y_pred = model.predict_classes(X_test).reshape(-1)
    print(Y_pred)
    print(Y_test)
    print("Acc : {} ".format(accuracy_score(Y_test, Y_pred)))
    print("Precision: {}".format(precision_score(Y_test, Y_pred)))
    print("Recall {}".format(recall_score(Y_test, Y_pred)))
    print("F1 {}".format(f1_score(Y_test, Y_pred)))
    connection.close()


if __name__ == "__main__":
    main()
