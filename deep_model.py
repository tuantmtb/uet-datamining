import pymysql
from keras.models import  Model,Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras
import numpy as np
import pickle
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,SVR
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

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
model_save_path="spam_model.h5"

def get_data(connection):
    with connection.cursor() as cursor:
        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='SPAM' LIMIT %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TRAIN_DOCS // 2))

        train_spam_docs = cursor.fetchall()
        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='NON-SPAM' LIMIT %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TRAIN_DOCS - len(train_spam_docs)))

        train_ham_docs = cursor.fetchall()

        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='SPAM' LIMIT %s OFFSET %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TEST_DOCS // 2, len(train_spam_docs)))

        test_spam_docs = cursor.fetchall()
        sql = "SELECT * FROM `extracted` WHERE `lang`=%s AND `label`='NON-SPAM' LIMIT %s OFFSET %s"
        cursor.execute(sql, (LANG, NUMBER_OF_TEST_DOCS - len(test_spam_docs), len(train_ham_docs)))

        test_ham_docs = cursor.fetchall()

        test_spam_docs.extend(test_ham_docs)

        train_spam_docs.extend(train_ham_docs)

        return train_spam_docs, test_spam_docs



def extract_data(train_docs,test_docs):
    contents = [doc['tokenize'] for doc in train_docs]
    contents_test = [doc['tokenize'] for doc in test_docs]
    merge_contents = contents + contents_test
    
    targets= [doc['label'] for doc in train_docs]
    targets_test = [doc['label'] for doc in test_docs]
    tok = TfidfVectorizer(max_features=NUMBER_OF_TERMS,smooth_idf=True,analyzer='word',sublinear_tf=True,
                         tokenizer=lambda doc: doc.lower().split(" "),ngram_range=(2, 6))
    tok.fit(merge_contents)

    sample_texts = tok.transform (contents).todense()
    targets = np.array([1 if x == 'SPAM' else 0 for x in targets],dtype=np.int32)

    sample_texts_test = tok.transform(contents_test).todense()
    targets_test = np.array([1 if x == 'SPAM' else 0 for x in targets_test],dtype=np.int32)

    return sample_texts,targets,sample_texts_test,targets_test

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

def check_model(model,x,y,epochs=4):
    checkpointer = ModelCheckpoint(filepath=model_save_path,
                                    verbose=1,
                                    save_best_only=True)    

    history = model.fit(x,y,batch_size=32,epochs=epochs,verbose=1,shuffle=True,validation_split=0.25,
              callbacks=[checkpointer]).history
              
    return history

def main():
    train_docs , test_docs = get_data(connection)
    X_train, Y_train ,  X_test, Y_test = extract_data(train_docs,test_docs)

    model = SVC()

    # train model
    model.fit(X_train, Y_train)
    # model = build_model()
    # his = check_model(model,X_train,Y_train,50)
    Y_pred = model.predict(X_test)
    # Y_pred = model.predict_classes(X_test).reshape(-1)
    print(Y_pred)
    print(Y_test)
    print("Acc : {} ".format(accuracy_score(Y_test,Y_pred)))
    print("Precision: {}".format(precision_score(Y_test,Y_pred)))
    print("Recall {}".format(recall_score(Y_test,Y_pred)))
    print("F1 {}".format(f1_score(Y_test,Y_pred)))
    connection.close()

if __name__ == "__main__":
    main()