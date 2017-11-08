# -*- coding: utf-8 -*-
from __future__ import division
import pymysql.cursors
import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
NUMBER_OF_TEST_DOCS = 333
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
        sql = "SELECT * FROM `extracted` WHERE `lang`=%s LIMIT %s"
        cursor.execute(sql, ('vi', NUMBER_OF_TRAIN_DOCS))

        train_docs = cursor.fetchall()

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s LIMIT %s OFFSET %s"
        cursor.execute(sql, ('vi', NUMBER_OF_TEST_DOCS, NUMBER_OF_TRAIN_DOCS))

        test_docs = cursor.fetchall()

        # docs = [
        #     {
        #         'lang': 'vi',
        #         'tokenize': 'Tf- term frequency : dùng để ước lượng tần xuất xuất hiện của từ trong văn bản. Tuy nhiên với mỗi văn bản thì có độ dài khác nhau, vì thế số lần xuất hiện của từ có thể nhiều hơn . Vì vậy số lần xuất hiện của từ sẽ được chia độ dài của văn bản (tổng số từ trong văn bản đó)'
        #     },
        #     {
        #         'lang': 'vi',
        #         'tokenize': 'IDF- Inverse Document Frequency: dùng để ước lượng mức độ quan trọng của từ đó như thế nào . Khi tính tần số xuất hiện tf thì các từ đều được coi là quan trọng như nhau. Tuy nhiên có một số từ thường được được sử dụng nhiều nhưng không quan trọng để thể hiện ý nghĩa của đoạn văn , ví dụ :'
        #     }
        # ]

        dict = make_dictionary(train_docs)

        print dict

        features_matrix, labels = extract_features(train_docs, dict)
        test_features_matrix, test_labels = extract_features(test_docs, dict)

        model = RandomForestClassifier(n_estimators=30)

        print features_matrix.shape
        print labels.shape

        print "Trainning model."

        # train model
        model.fit(features_matrix, labels)

        predicted_labels = model.predict(test_features_matrix)

        print "FINISHED classifying. accuracy score : "
        print accuracy_score(test_labels, predicted_labels)

finally:
    connection.close()
