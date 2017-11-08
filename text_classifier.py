# -*- coding: utf-8 -*-
from __future__ import division
import pymysql.cursors
import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

NUMBER_OF_TRAIN_DOCS = 600
NUMBER_OF_TEST_DOCS = 200
NUMBER_OF_TERMS = 1000


def get_content(doc):
    if doc['lang'] == 'vi':
        content = doc['tokenize']
    else:
        content = doc['data']

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
    list_to_remove = dictionary.keys()

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
    training_labels = np.zeros(len(docs))

    idf_dict = get_idf_dictionary(docs, dictionary)

    for docId, doc in enumerate(docs):
        content = get_content(doc)
        words = content.split()

        if doc['label'] == 'SPAM':
            training_labels[docId] = 1
        else:
            training_labels[docId] = 0

        for word in words:
            for termId, d in enumerate(dictionary):
                if d[0] == word:
                    tf_idf = calculate_tf_idf(word, words, len(docs), idf_dict)

                    features_matrix[docId, termId] = tf_idf

    return features_matrix, training_labels


try:
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='SPAM' LIMIT %s"
        cursor.execute(sql, ('vi', NUMBER_OF_TRAIN_DOCS // 2))

        train_spam_docs = cursor.fetchall()

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='NON-SPAM' LIMIT %s"
        cursor.execute(sql, ('vi', NUMBER_OF_TRAIN_DOCS - len(train_spam_docs)))

        train_ham_docs = cursor.fetchall()

        train_docs = np.concatenate((train_spam_docs, train_ham_docs), axis=0)
        random.shuffle(train_docs)

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='SPAM' LIMIT %s OFFSET %s"
        cursor.execute(sql, ('vi', NUMBER_OF_TEST_DOCS // 2, len(train_spam_docs)))

        test_spam_docs = cursor.fetchall()

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='NON-SPAM' LIMIT %s OFFSET %s"
        cursor.execute(sql, ('vi', NUMBER_OF_TEST_DOCS - len(test_spam_docs), len(train_ham_docs)))

        test_ham_docs = cursor.fetchall()

        test_docs = np.concatenate((test_ham_docs, test_spam_docs), axis=0)
        random.shuffle(test_docs)

        dict = make_dictionary(train_docs)

        features_matrix, labels = extract_features(train_docs, dict)
        test_features_matrix, test_labels = extract_features(test_docs, dict)

        model = RandomForestClassifier(n_estimators=3)

        print "Trainning model."

        # train model
        model.fit(features_matrix, labels)

        predicted_labels = model.predict(test_features_matrix)

        print "FINISHED classifying. accuracy score : "
        print accuracy_score(test_labels, predicted_labels)

finally:
    connection.close()
