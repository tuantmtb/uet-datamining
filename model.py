# -*- coding: utf-8 -*-
from __future__ import division
import pymysql.cursors
import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.externals import joblib
import pickle
import random

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


def get_content(doc):
    if doc['lang'] == 'vi':
        content = doc['tokenize']
    else:
        content = doc['normalized']

    return content


def get_idf_dictionary(docs, dictionary):
    # idf_matrix = np.zeros((len(docs), len(dictionary)))

    dict = {}

    for termId, d in enumerate(dictionary):
        ndocs = 0
        term = d[0]

        for docId, doc in enumerate(docs):
            content = get_content(doc)

            if term in content:
                ndocs += 1

        dict[term] = ndocs

    return dict


def make_dictionary(docs):
    all_words = []

    for doc in docs:
        content = get_content(doc)
        words = content.split()
        all_words += words

    dictionary = collections.Counter(all_words)
    list_to_remove = dictionary.copy().keys()

    for item in list_to_remove:
        # remove if numerical
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(NUMBER_OF_TERMS)

    return dictionary


def calculate_tf_idf(word, words, ndocs, idf_dict):
    tf = words.count(word) / len(words)
    idf = np.log(ndocs / idf_dict[word])

    return tf * idf


def extract_features(docs, dictionary):
    features_matrix = np.zeros((len(docs), NUMBER_OF_TERMS))

    idf_dict = get_idf_dictionary(docs, dictionary)

    for docId, doc in enumerate(docs):
        content = get_content(doc)
        words = content.split()

        for word in words:
            for termId, d in enumerate(dictionary):
                if d[0] == word:
                    tf_idf = calculate_tf_idf(word, words, len(docs), idf_dict)

                    features_matrix[docId, termId] = tf_idf

    return features_matrix


def extract_labels(docs):
    training_labels = np.zeros(len(docs))

    for docId, doc in enumerate(docs):
        if doc['label'] == 'SPAM':
            training_labels[docId] = 1
        else:
            training_labels[docId] = 0

    return training_labels


def get_data(connection):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='SPAM' LIMIT %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TRAIN_DOCS // 2))

        train_spam_docs = cursor.fetchall()

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='NON-SPAM' LIMIT %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TRAIN_DOCS - len(train_spam_docs)))

        train_ham_docs = cursor.fetchall()

        train_docs = np.concatenate((train_spam_docs, train_ham_docs), axis=0)
        random.shuffle(train_docs)

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='SPAM' LIMIT %s OFFSET %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TEST_DOCS // 2, len(train_spam_docs)))

        test_spam_docs = cursor.fetchall()

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='NON-SPAM' LIMIT %s OFFSET %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TEST_DOCS - len(test_spam_docs), len(train_ham_docs)))

        test_ham_docs = cursor.fetchall()

        test_docs = np.concatenate((test_ham_docs, test_spam_docs), axis=0)
        random.shuffle(test_docs)

        return train_docs, test_docs

def train():
    try:

        print ("Getting data...")

        train_docs, test_docs = get_data(connection)

        print ("Processing data...")

        dict = make_dictionary(train_docs)

        with open('dict.pkl', 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        features_matrix = extract_features(train_docs, dict)
        labels = extract_labels(train_docs)

        # with open('data_x.pkl', 'wb') as handle:
        #     pickle.dump(features_matrix,handle,pickle.HIGHEST_PROTOCOL)
        # with open('data_y.pkl', 'wb') as handle:
        #     pickle.dump(labels,handle,pickle.HIGHEST_PROTOCOL)

        test_features_matrix = extract_features(test_docs, dict)
        test_labels = extract_labels(test_docs)
        # with open('test_data_x.pkl', 'wb') as handle:
        #     pickle.dump(test_features_matrix,handle,pickle.HIGHEST_PROTOCOL)
        # with open('test_data_y.pkl', 'wb') as handle:
        #     pickle.dump(test_labels,handle,pickle.HIGHEST_PROTOCOL)

        print ("Training model...")

        model = GaussianNB()

        # train model
        model.fit(features_matrix, labels)

        joblib.dump(model, LANG + '_model.pkl')

        print ("Testing...")

        predicted_labels = model.predict(test_features_matrix)

        print( "FINISHED classifying. accuracy score : ")
        print("Acc : {} ".format(accuracy_score(test_labels,predicted_labels)))
        print("Precision: {}".format(precision_score(test_labels,predicted_labels)))
        print("Recall {}".format(recall_score(test_labels,predicted_labels)))
        print("F1 {}".format(f1_score(test_labels,predicted_labels)))        
    finally:
        connection.close()


def predict(doc):
    clf = joblib.load('filename.pkl')

    with open('dict.pkl', 'rb') as handle:
        dict = pickle.load(handle)
        feature_matrix = extract_features([doc], dict)

        label = clf.predict(feature_matrix)[0]

        print (label)


