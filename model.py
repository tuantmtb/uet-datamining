# -*- coding: utf-8 -*-
from __future__ import division
import pymysql.cursors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib
import model_helpers
from config import *

connection = pymysql.connect(host=DB_HOST,
                             user=DB_USER,
                             password=DB_PASSWORD,
                             db=DB_NAME,
                             port=DB_PORT,
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor)


def train():
    try:

        print("Getting data...")

        train_docs, test_docs = model_helpers.get_data(connection, LANG, NUMBER_OF_TRAIN_DOCS, NUMBER_OF_TEST_DOCS)

        print("Processing data...")

        model_helpers.train_tf_idf(train_docs + test_docs, NUMBER_OF_TERMS, LANG)
        features_matrix, labels = model_helpers.extract_data(train_docs, LANG)
        test_features_matrix, test_labels = model_helpers.extract_data(test_docs, LANG)

        print("Training model...")

        model = RandomForestClassifier(n_estimators=10)

        # train model
        model.fit(features_matrix, labels)

        joblib.dump(model, "model-data/" + LANG + '_model.pkl')

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
    clf = joblib.load("model-data/" + doc['lang'] + '_model.pkl')

    feature_matrix, _ = model_helpers.extract_data([doc], doc['lang'])

    label = clf.predict(feature_matrix)[0]

    return label


if __name__ == "__main__":
    train()
