import csv
import os
import pickle
import re
from string import punctuation
import fasttext
import nltk
import numpy as np
import pandas as pd
import pymorphy2
import spacy
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer
from gensim.models import FastText
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
fasttext.FastText.eprint = lambda x: None
tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer, lemmatize=True)


def csv_reader(file_obj):
    """
    Read a csv file
    """

    reader = csv.reader(file_obj)
    data = []
    for row in reader:
        row = str(" ".join(row))
        row = row.replace("#", "hashtag")
        if row != "":
            data.append(row)
    data[0] = "comment;toxic;PER;LOC;ORG;positive;negative;neutral;speech;skip;rude"
    return data


def csv_dict_writer(path, fieldnames, data):
    """
    Writes a CSV file using DictWriter
    """
    with open(path, "w", newline='', encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, delimiter=';', fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"


def preprocessing_data(text):
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    stop_words = set(stopwords.words('russian') + list(punctuation))  # список стоп-слов
    morph = pymorphy2.MorphAnalyzer()  # для лемматизации
    text = text.lower()  # приведение слов к строчным
    text = re.sub(patterns, ' ', text, flags=re.MULTILINE)  # удаляем знаки пунктуации
    tokens = word_tokenize(text)  # разделяем слова на отдельные токены
    text = [word for word in tokens if word not in stop_words]  # удаляем стоп-слова
    text = [morph.normal_forms(word.strip())[0] for word in text]  # производим стемминг
    text = ' '.join(text)
    return text


def write_pickle(file, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(file, f)


def read_pickle(name):
    print(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.dirname(os.path.abspath(__file__)) + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def preprocess(input_data):
    train_db = pd.read_csv(input_data, sep=",", error_bad_lines=False)  # получение датасета
    print("Препроцессинг")
    train_db['comment'] = train_db['comment'].map(preprocessing_data)  # подготовка данных к обработке
    return train_db


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # если текст пустой, мы должны возвращать вектор нулей
        # с такой же размерностью, как и остальные вектора
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)


class MyTokenizer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


def train_word2vec(tokens):
    w2v_model = Word2Vec(min_count=5, window=10, size=150, negative=10,
                         alpha=0.03, min_alpha=0.0007, sample=6e-5, sg=0)
    w2v_model.build_vocab(tokens)
    print(w2v_model.corpus_count)
    w2v_model.train(tokens, total_examples=w2v_model.corpus_count, epochs=300, report_delay=1)
    w2v_model.init_sims(replace=True)
    write_pickle(w2v_model, 'w2v_model5')
    return w2v_model


def train_fasttext(tokens):
    ft_model = FastText(min_count=10, window=5, size=150, negative=10, alpha=0.03, min_alpha=0.0007, sample=6e-5, sg=0)
    ft_model.build_vocab(tokens)
    print(ft_model.corpus_count)
    ft_model.train(tokens, total_examples=ft_model.corpus_count, epochs=300, report_delay=1)
    ft_model.init_sims(replace=True)
    write_pickle(ft_model, 'ft_model2')
    return ft_model


# определят соотношение матов к количеству текста
def rude_feature_extraction(comment):
    with open("rude_words.txt", "r", encoding="utf-8") as f_txt:
        r_w = f_txt.readline()
    r_w = r_w.replace(" ", "")
    r_w = r_w.split(",")
    lemmy = preprocessing_data(comment).split(" ")
    rude_words_counter = 0
    for lemma in lemmy:
        if r_w.count(lemma) != 0:
            rude_words_counter += 1
    return rude_words_counter / len(lemmy)


# потенциально будет определять эмоциональный окрас
def some_spicy_features_extraction(comment):
    nlp = spacy.load("ru_core_news_md")
    res1 = nlp(comment)
    per_c = "0"
    loc_c = "0"
    org_c = "0"
    for ent in res1.ents:
        if len(ent) != 0:
            if ent.label_ == "PER":
                per_c = "1"
            if ent.label_ == "LOC":
                loc_c = "1"
            if ent.label_ == "ORG":
                org_c = "1"
    res = model.predict([comment], k=5)[0]
    pos_c = str(res["positive"])
    neg_c = str(res["negative"])
    neu_c = str(res["neutral"])
    sp_c = str(res["speech"])
    sk_c = str(res["skip"])
    return per_c, loc_c, org_c, pos_c, neg_c, neu_c, sp_c, sk_c


def training_data(input_data):
    train_db = pd.read_csv(input_data)  # получение датасета
    # train_db = train_db[:100]
    # print("Препроцессинг")
    # train_db['comment'] = train_db['comment'].map(preprocessing_data)  # подготовка данных к обработке
    # разделение на тренировочную и тестовую выборку
    # data = preprocess(input_data)
    train_db = read_pickle("preprocessed_words3")
    train_db = train_db.dropna()
    X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
                                                        test_size=0.33,
                                                        random_state=42)
    print("Векторизация")
    # формирование словаря
    # tokens = train_db["comment"].values.astype('U')
    # tokens = [token.split(" ") for token in tokens]
    # train_word2vec(tokens)
    model_vect = read_pickle("w2v_model5")
    mev = MeanEmbeddingVectorizer(model_vect)
    X_train_counts = mev.fit_transform(X_train)
    X_train_counts.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(X_train_counts)
    X_train_counts = scaler.transform(X_train_counts)
    X_train_counts = sparse.csr_matrix(X_train_counts)

    print("Обучение SVC классификатора")
    # тренировка классификтора
    # model = SVC(kernel='linear', probability=True, C=0.1).fit(X_train_counts, y_train)
    # write_pickle(model, 'modelSVCw2v')
    model = read_pickle('modelSVCw2v')

    # формирование тренировочной выборки
    X_test_counts = mev.fit_transform(X_test)
    X_test_counts.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(X_test_counts)
    X_test_counts = scaler.transform(X_test_counts)
    X_test_counts = sparse.csr_matrix(X_test_counts)

    print("Оценка точности SVC классификатора")
    # оценка точности классификатора
    predicted = model.predict(X_test_counts)
    acc = np.mean(predicted == y_test)
    print("Точность: ", acc)

    report = classification_report(y_test, model.predict(X_test_counts), target_names=['untoxic', 'toxic'])
    print(report)


def classifier(messages):
    clear_message = map(preprocessing_data, messages)
    model_vect = read_pickle('w2v_model5')
    mev = MeanEmbeddingVectorizer(model_vect)
    X_train_counts = mev.fit_transform(clear_message)
    X_train_counts.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(X_train_counts)
    X_train_counts = scaler.transform(X_train_counts)
    X_train_counts = sparse.csr_matrix(X_train_counts)

    model = read_pickle('modelSVCw2v')
    predicted = model.predict_proba(X_train_counts)
    return zip(messages, predicted)


if __name__ == '__main__':
    messages = ["Верблюдов-то за что? Дебилы, бл...",
                "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. Если бы хохлов не было, кисель их бы придумал.",
                "Какой чудесный день!",
                "ты вообще отстойный, фу таким быть",
                "Световые столбы в 2 ночи..."]
    input_data = 'labeled_ru_ds.csv'  # для новой прогонки
    # training_data(input_data)
    res = some_spicy_features_extraction(messages[0])
    print(res)
    labeled_messages = classifier(messages)
    print(list(labeled_messages))
    # counter = 0
    # for comment, toxic in labeled_messages:
    #     print('%r => %s' % (comment, toxic))
    #     if (counter == 10):
    #         break
    #     counter += 1
