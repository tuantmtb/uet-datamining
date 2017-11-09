from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.externals import joblib


def get_data(connection, LANG, NUMBER_OF_TRAIN_DOCS, NUMBER_OF_TEST_DOCS):
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


def extract_data(docs, lang):
    tok = joblib.load(lang + "_tok.pkl")
    contents = [doc['tokenize'] for doc in docs]

    targets = [doc['label'] for doc in docs]

    sample_texts = tok.transform(contents).todense()
    targets = np.array([1 if x == 'SPAM' else 0 for x in targets], dtype=np.int32)

    return sample_texts, targets


def tokenize(doc):
    return doc.lower().split()


def train_tf_idf(train_docs, number_of_terms, lang):
    train_contents = [doc['tokenize'] for doc in train_docs]

    tok = TfidfVectorizer(max_features=number_of_terms, smooth_idf=True, analyzer='word', sublinear_tf=True,
                          tokenizer=tokenize, ngram_range=(2, 6))

    tok.fit(train_contents)

    joblib.dump(tok, lang + '_tok.pkl')
