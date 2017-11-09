# -*- coding: utf-8 -*-
from __future__ import division
import pymysql.cursors
import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
import pickle
import model_helpers

import json
import numpy as np

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


def train():
    try:

        print("Getting data...")

        train_docs, test_docs = model_helpers.get_data(connection, LANG, NUMBER_OF_TRAIN_DOCS, NUMBER_OF_TEST_DOCS)

        print("Processing data...")

        model_helpers.train_tf_idf(train_docs, NUMBER_OF_TERMS, LANG)
        features_matrix, labels = model_helpers.extract_data(train_docs, LANG)
        test_features_matrix, test_labels = model_helpers.extract_data(test_docs, LANG)

        print("Training model...")

        model = RandomForestClassifier(n_estimators=10)

        # train model
        model.fit(features_matrix, labels)

        joblib.dump(model, LANG + '_model.pkl')

        print("Testing...")

        predicted_labels = model.predict(test_features_matrix)

        print("FINISHED classifying. accuracy score : ")
        print("Acc : {} ".format(accuracy_score(test_labels, predicted_labels)))
        print("Precision: {}".format(precision_score(test_labels, predicted_labels)))
        print("Recall {}".format(recall_score(test_labels, predicted_labels)))
        print("F1 {}".format(f1_score(test_labels, predicted_labels)))
    finally:
        connection.close()


def predict(doc):
    clf = joblib.load('filename.pkl')

    with open('dict.pkl', 'rb') as handle:
        dict = pickle.load(handle)
        feature_matrix = ([doc], dict)

        label = clf.predict(feature_matrix)[0]

        print(label)


if __name__ == "__main__":
    train()
