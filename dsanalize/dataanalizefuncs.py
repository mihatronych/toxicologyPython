import csv
import pickle
import re
from collections import defaultdict
from string import punctuation
import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymorphy2
import seaborn as sns
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.tokenization import RegexTokenizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from gensim.models import FastText
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import TransformerMixin
import nltk
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss, accuracy_score
from gensim.models import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def preprocess(input_data):
    train_db = pd.read_csv(input_data, sep=",", error_bad_lines=False)  # получение датасета
    print("Препроцессинг")
    train_db['comment'] = train_db['comment'].map(preprocessing_data)  # подготовка данных к обработке
    return train_db


def word_to_vect(data):
    print("Векторизация")
    train_db = data  # предобработка
    # разделение на тренировочную и тестовую выборку
    # X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
    #                                                    test_size=0.2,
    #                                                    random_state=42)
    # формирование словаря
    count_vect = CountVectorizer(max_features=500)
    X_train_counts = count_vect.fit_transform(train_db['comment'].values.astype('U'))
    write_pickle(count_vect, 'count_vect')
    return X_train_counts, count_vect


def tf_idf(data):
    print("формирование TF-IDF")
    X_train_counts = data
    # формирование TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    write_pickle(tfidf_transformer, 'tfidf_transformer')
    return X_train_tfidf, tfidf_transformer


def word_freq_func(data):
    word_freq = defaultdict(int)
    for tokens in data.iloc[:]:
        for token in tokens:
            word_freq[token] += 1
    return word_freq


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


def evaluate_features(X, y, clf=None):
    """Основная вспомогательная функция для проверки эффективности поступивших признаков в модель машинного обучения
    Выводит отчет по всем данным
    Аргументы:
        X (array-like): Массив признаков. Форма (n_samples, n_features)
        y (array-like): Массив меток. Форма (n_samples,)
        clf: Используемый классификатор. При отсутствии используется классификация логарифмической регрессией"""
    if clf is None:
        clf = LogisticRegression()
    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                               n_jobs=-1, method='predict_proba', verbose=2)
    write_pickle(clf, "class_model")
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    report = classification_report(y, preds, target_names=['untoxic', 'toxic'], digits=3)
    print(report)
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    tn1, fp1, fn1, tp1 = tn, fp, fn, tp
    print(tn, fp, fn, tp)
    print("Accuracy: ", ((tp + tn) / (tp + tn + fp + fn)))
    print("Precision: ", ((tp) / (tp + fp)))
    print("Recall: ", ((tp) / (tp + fn)))
    tp1 = tn
    fn1 = fp
    fp1 = fn
    tn1 = tp
    print(tn1, fp1, fn1, tp1)
    print("Accuracy: ", ((tp1 + tn1) / (tp1 + tn1 + fp1 + fn1)))
    print("Precision: ", ((tp1) / (tp1 + fp1)))
    print("Recall: ", ((tp1) / (tp1 + fn1)))
    report = classification_report(y, preds, target_names=['untoxic', 'toxic'], digits=3)
    print(report)


class w2vTransformer(TransformerMixin):
    """
    Wrapper class for running word2vec into pipelines and FeatureUnions
    """

    def __init__(self, word2vec, **kwargs):
        self.word2vec = word2vec
        self.kwargs = kwargs
        self.dim = word2vec.vector_size

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words.split(" ") if w in self.word2vec.wv.vocab.keys()]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


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


def train_doc2vec(data):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
    model = Doc2Vec(documents, vector_size=10, workers=8, epochs=300)
    write_pickle(model, 'd2v_model3')


def tsne_scatterplot(model, word, list_names):
    """Plot in seaborn the results from the t-SNE dimensionality reduction
    algorithm of the vectors of a query word,
    its list of most similar words, and a list of words."""
    vectors_words = [model.wv.word_vec(word)]
    word_labels = [word]
    color_list = ['red']

    close_words = model.wv.most_similar(word, topn=500)
    for wrd_score in close_words:
        wrd_vector = model.wv.word_vec(wrd_score[0])
        vectors_words.append(wrd_vector)
        word_labels.append(wrd_score[0])
        color_list.append('blue')

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.word_vec(wrd)
        vectors_words.append(wrd_vector)
        word_labels.append(wrd)
        color_list.append('green')

    # t-SNE reduction
    Y = (TSNE(n_components=2, random_state=0, perplexity=15, init="pca")
         .fit_transform(vectors_words))
    # Sets everything up to plot
    df = pd.DataFrame({"x": [x for x in Y[:, 0]],
                       "y": [y for y in Y[:, 1]],
                       "words": word_labels,
                       "color": color_list})
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={"s": 40,
                                  "facecolors": df["color"]}
                     )
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df["y"][line],
                " " + df["words"][line].title(),
                horizontalalignment="left",
                verticalalignment="bottom", size="medium",
                color=df["color"][line],
                weight="normal"
                ).set_size(7)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)
    plt.title('t-SNE visualization for {}'.format(word.title()))
    return plt


def some_spicy_features_extraction(input_data, nlp):  # version pymorphy 0.9
    with open(input_data, "r", encoding="utf-8") as f_csv:
        data = csv_reader(f_csv)
    loc_data = data[1:len(data)]
    loc_data = [i.split(";") for i in loc_data]
    for k in range(len(loc_data)):
        row = loc_data[k]
        res = nlp(row[0])
        for ent in res.ents:
            if len(ent) != 0:
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                if ent.label_ == "PER":
                    row[2] = "1"
                if ent.label_ == "LOC":
                    row[3] = "1"
                if ent.label_ == "ORG":
                    row[4] = "1"
        loc_data[k] = row
        res = model.predict([row[0]], k=5)[0]
        row[5] = str(res["positive"])
        row[6] = str(res["negative"])
        row[7] = str(res["neutral"])
        row[8] = str(res["speech"])
        row[9] = str(res["skip"])
        loc_data[k] = row
    loc_data = [";".join(i) for i in loc_data]
    data[1:len(data)] = loc_data
    data = [row.split(";") for row in data]
    my_list = []
    fieldnames = data[0]
    for values in data[1:]:
        inner_dict = dict(zip(fieldnames, values))
        my_list.append(inner_dict)
    path = "vk_comments_DS2.csv"
    csv_dict_writer(path, fieldnames, my_list)


def rude_feature_extraction(input_data):
    with open(input_data, "r", encoding="utf-8") as f_csv:
        data = csv_reader(f_csv)
    loc_data = data[1:len(data)]
    loc_data = [i.split(";") for i in loc_data]
    with open("rude_words.txt", "r", encoding="utf-8") as f_txt:
        r_w = f_txt.readline()
    r_w = r_w.replace(" ", "")
    r_w = r_w.split(",")
    for k in range(len(loc_data)):
        row = loc_data[k]
        lemmy = preprocessing_data(row[0]).split(" ")
        for lemma in lemmy:
            if r_w.count(lemma) != 0:
                loc_data[k][10] = "1"
    data[1:len(data)] = loc_data
    my_list = []
    fieldnames = data[0].split(";")
    for values in data[1:]:
        inner_dict = dict(zip(fieldnames, values))
        my_list.append(inner_dict)
    path = "vk_comments_DS3.csv"
    csv_dict_writer(path, fieldnames, my_list)


def coeff_corel(data_x, data_y):
    coef, p = pearsonr(data_x, data_y)
    return coef, p


class NBFeaturer(BaseEstimator, ClassifierMixin):
    """from https://www.kaggle.com/sermakarevich"""

    def __init__(self, alpha):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

    def fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)


def train_model(input_data, model_vect, model_name, cv_tfidf=False):
    train_db = input_data[:14000]
    X = train_db['comment']
    y = train_db['toxic']
    if model_name == "SVC":
        # тренировка классификтора
        model = SVC(kernel='linear', probability=True, C=0.1)

    if model_name == "OVRSVC":  # Пока не пашет
        # тренировка классификтора
        model = OneVsRestClassifier(SVC(kernel='linear', probability=True, C=0.1))

    if model_name == "MNB":
        # тренировка классификтора
        model = MNB(alpha=1)

    if model_name == "NBSVM":
        nbf = NBFeaturer(alpha=1)
        model = SVC(kernel='linear', probability=True, C=1)  # One-vs.-rest balanced category with 1:1:1

        p = pipeline = Pipeline([
            ('nbf', nbf),
            ('lr', model)
        ])
        model = p

    if model_name == "RFCCV":
        if type(model_vect).__name__ == "CountVectorizer":
            # формирование словаря
            X_train_counts = model_vect.transform(X)
            X_train_counts.shape[0]
            # scaler = StandardScaler(with_mean=False)
            # scaler.fit(X_train_counts)
            # X_train_counts = scaler.transform(X_train_counts)
            if cv_tfidf == True:
                X_train_counts = TfidfTransformer().fit_transform(X_train_counts)
            if model_name == "MNB":
                scaler = MaxAbsScaler()
                scaler.fit(X_train_counts)
                X_train_counts = scaler.transform(X_train_counts)

        if type(model_vect).__name__ == "Word2Vec":
            # формирование словаря
            mev = MeanEmbeddingVectorizer(model_vect)
            X_train_counts = mev.fit_transform(X)
            X_train_counts.shape[0]
            X_train_counts = sparse.csr_matrix(X_train_counts)
            if model_name == "MNB":
                scaler = MaxAbsScaler()
                scaler.fit(X_train_counts)
                X_train_counts = scaler.transform(X_train_counts)

        if type(model_vect).__name__ == "Doc2Vec":
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
            X_train_counts = [model_vect.infer_vector(str(x.words).split(" ")) for x in documents]
            if model_name == "MNB":
                scaler = MinMaxScaler()
                scaler.fit(X_train_counts)
                X_train_counts = scaler.transform(X_train_counts)
                print("AAAAAAA")
                print(X_train_counts)

        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        max_features = ['log2', 'sqrt']
        max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
        min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
        min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
        bootstrap = [True, False]
        param_dist = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}
        rfc_2 = RandomForestClassifier()
        rs = RandomizedSearchCV(rfc_2,
                                param_dist,
                                n_iter=100,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                random_state=0)
        rs.fit(X_train_counts, y)
        print(rs.best_params_)
        return
        # n_estimators=600, min_samples_split=23, min_samples_leaf=2, max_features='sqrt', max_depth=15, bootstrap=False
        # n_estimators=700, min_samples_split=2, min_samples_leaf=2, max_features='log2', max_depth=11, bootstrap=True
        # n_estimators=400, min_samples_split=23, min_samples_leaf=2, max_features='sqrt', max_depth=14, bootstrap=False
    if model_name == "RFC":
        model = RandomForestClassifier(n_estimators=600, min_samples_split=23, min_samples_leaf=2, max_features='sqrt',
                                       max_depth=15, bootstrap=False)

    print("Векторизация")
    if type(model_vect).__name__ == "CountVectorizer":
        # формирование словаря
        X_train_counts = model_vect.transform(X)
        X_train_counts.shape[0]
        # scaler = StandardScaler(with_mean=False)
        # scaler.fit(X_train_counts)
        # X_train_counts = scaler.transform(X_train_counts)
        if cv_tfidf == True:
            X_train_counts = TfidfTransformer().fit_transform(X_train_counts)
        if model_name == "MNB":
            scaler = MaxAbsScaler()
            scaler.fit(X_train_counts)
            X_train_counts = scaler.transform(X_train_counts)

    if type(model_vect).__name__ == "Word2Vec":
        # формирование словаря
        mev = MeanEmbeddingVectorizer(model_vect)
        X_train_counts = mev.fit_transform(X)
        X_train_counts.shape[0]
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(X_train_counts)
        X_train_counts = scaler.transform(X_train_counts)
        X_train_counts = sparse.csr_matrix(X_train_counts)
        # X_train_counts = sparse.csr_matrix(X_train_counts)
        # if model_name == "MNB":
        #    scaler = MaxAbsScaler()
        #    scaler.fit(X_train_counts)
        #    X_train_counts = scaler.transform(X_train_counts)

    if type(model_vect).__name__ == "Doc2Vec":
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        X_train_counts = [model_vect.infer_vector(str(x.words).split(" ")) for x in documents]
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(X_train_counts)
        X_train_counts = scaler.transform(X_train_counts)
        X_train_counts = sparse.csr_matrix(X_train_counts)
        # if model_name == "MNB":
        #    scaler = MinMaxScaler(feature_range=(0,100))
        #    scaler.fit(X_train_counts)
        #    X_train_counts = scaler.transform(X_train_counts)

    if type(model_vect).__name__ == "FastText":
        # формирование словаря
        mev = MeanEmbeddingVectorizer(model_vect)
        X_train_counts = mev.fit_transform(X)
        X_train_counts.shape[0]
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaler.fit(X_train_counts)
        X_train_counts = scaler.transform(X_train_counts)
        X_train_counts = sparse.csr_matrix(X_train_counts)
    print(type(X_train_counts).__name__)
    evaluate_features(X_train_counts, y, model)


def train_svc_ft(input_data, model_vect):
    X_train, X_test, y_train, y_test = train_test_split(input_data['comment'].values.astype('U'), input_data['toxic'],
                                                        test_size=0.20,
                                                        random_state=42)
    # model = read_pickle("class_model")
    mev = MeanEmbeddingVectorizer(model_vect)
    X_train_counts = mev.fit_transform(X_train)
    # X_train_counts.shape[0]
    # scaler = MinMaxScaler(feature_range=(0, 100))
    # scaler.fit(X_train_counts)
    # X_train_counts = scaler.transform(X_train_counts)
    # X_train_counts = sparse.csr_matrix(X_train_counts)
    # model = model.fit(X_train_counts, y_train)
    # write_pickle(model, 'modelSVC')
    model = read_pickle("modelSVC")
    X_new_counts = mev.transform(X_test)
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(X_new_counts)
    X_new_counts = scaler.transform(X_new_counts)
    X_new_counts = sparse.csr_matrix(X_new_counts)
    report = classification_report(y_test, model.predict(X_new_counts), target_names=['untoxic', 'toxic'])
    print(report)

    dif_data = read_pickle("preprocessed_words2")
    X_counts = mev.transform(dif_data["comment"][:5000])
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(X_counts)
    X_counts = scaler.transform(X_counts)
    X_counts = sparse.csr_matrix(X_counts)
    report = classification_report(dif_data["toxic"][:5000], model.predict(X_counts), target_names=['untoxic', 'toxic'])
    print("Другой набор данных")
    print(report)
    X_counts = mev.transform(dif_data["comment"])
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit(X_counts)
    X_counts = scaler.transform(X_counts)
    X_counts = sparse.csr_matrix(X_counts)
    res = model.predict(X_counts)
    final_matrix = np.array([np.array(dif_data["comment"]), np.array(res)])
    df = pd.DataFrame(final_matrix)
    df.to_excel("matrix_count_vect2.xlsx", header=True)


def train_svc(input_data, model_vect, model_name):
    train_db = input_data[:14000]

    X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
                                                        test_size=0.20,
                                                        random_state=42)

    if model_name == "SVC":
        # тренировка классификтора
        model = SVC(kernel='linear', probability=True, C=1)

    if model_name == "MNB":
        # тренировка классификтора
        model = MNB(alpha=1)

    if model_name == "NBSVM":
        nbf = NBFeaturer(alpha=1)
        model = LinearSVC(C=1, max_iter=10000)  # One-vs.-rest balanced category with 1:1:1

        p = pipeline = Pipeline([
            ('nbf', nbf),
            ('lr', model)
        ])
        model = p

    print("Векторизация")
    if type(model_vect).__name__ == "CountVectorizer":
        # формирование словаря
        X_train_counts = model_vect.transform(X_train)
        X_train_counts.shape
        if model_name == "MNB":
            scaler = MaxAbsScaler()
            scaler.fit(X_train_counts)
            X_train_counts = scaler.transform(X_train_counts)

    if type(model_vect).__name__ == "Word2Vec":
        # формирование словаря
        mev = MeanEmbeddingVectorizer(model_vect)
        X_train_counts = mev.fit_transform(X_train)
        X_train_counts.shape
        # evaluate_features(X_train_counts, y_train.values.ravel(),
        #                  SVC(kernel='linear', probability=True, C=0.1))
        if model_name == "MNB":
            scaler = MinMaxScaler()
            scaler.fit(X_train_counts)
            X_train_counts = scaler.transform(X_train_counts)

    if type(model_vect).__name__ == "Doc2Vec":
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
        X_train_counts = [model_vect.infer_vector(str(x.words).split(" ")) for x in documents]
        if model_name == "MNB":
            scaler = MinMaxScaler()
            scaler.fit(X_train_counts)
            X_train_counts = scaler.transform(X_train_counts)

    if type(model_vect).__name__ == "FastText":
        # формирование словаря
        mev = MeanEmbeddingVectorizer(model_vect)
        X_train_counts = mev.transform(X_train)
        X_train_counts.shape
        if model_name == "MNB":
            scaler = MinMaxScaler()
            scaler.fit(X_train_counts)
            X_train_counts = scaler.transform(X_train_counts)

    if model_name == "SVC":
        print("Обучение SVC классификатора")
        # тренировка классификтора
        model = model.fit(X_train_counts, y_train)
        write_pickle(model, 'modelSVC')

    if model_name == "MNB":
        print("Обучение MNB классификатора")
        # тренировка классификтора
        model = model.fit(X_train_counts, y_train)
        write_pickle(model, 'modelMNB')

    if model_name == "NBSVM":
        print("Обучение NBSVM классификатора")
        model = model.fit(X_train_counts, y_train)
        write_pickle(model, 'modelNBSVM')

    if type(model_vect).__name__ == "CountVectorizer":
        # формирование тренировочной выборки
        X_new_counts = model_vect.transform(X_test)
        # if model_name == "MNB":
        # X_new_counts = scaler.transform(X_new_counts)
        # print(X_new_counts)

    if type(model_vect).__name__ == "Word2Vec":
        # формирование тренировочной выборки
        X_new_counts = mev.fit_transform(X_test)
        # if model_name == "MNB":
        # X_new_counts  = scaler.transform(X_new_counts)
        # print(X_new_counts)

    if type(model_vect).__name__ == "Doc2Vec":
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_test)]
        X_new_counts = [model_vect.infer_vector(str(x.words).split(" ")) for x in documents]

        print(X_new_counts)

    if type(model_vect).__name__ == "FastText":
        # формирование тренировочной выборки
        X_new_counts = mev.transform(X_test)
        # if model_name == "MNB":
        # X_new_counts  = scaler.transform(X_new_counts)
        # print(X_new_counts)

    print("Оценка точности классификатора")
    # оценка точности классификатора

    acc = model.score(X_new_counts, y_test)
    print("Точность: ", acc)

    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_new_counts)).ravel()
    # print(tn, fp, fn, tp)
    # print("Accuracy: ", ((tp + tn) / (tp + tn + fp + fn)))
    # print("Precision: ", ((tp) / (tp + fp)))
    # print("Recall: ", ((tp) / (tp + fn)))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    report = classification_report(y_test, model.predict(X_new_counts), target_names=['untoxic', 'toxic'])
    print(report)


def document_vector(array_of_word_vectors):
    return array_of_word_vectors.mean(axis=0)


if __name__ == '__main__':
    # nlp = spacy.load("ru_core_news_lg")
    messages = ["Верблюдов-то за что? Дебилы, бл...",
                "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. Если бы хохлов не было, кисель их бы придумал.",
                "Какой чудесный день!",
                "ты вообще отстойный, фу таким быть"]
    input_data = 'labeled_ru_ds.csv'  # для новой прогонки
    # some_spicy_features_extraction(input_data, nlp) # функция доставания некоторых признаков:именованные,настроение
    # data = preprocess(input_data)
    # data = pd.read_csv("clear_data.csv", sep=",", error_bad_lines=False)
    # write_pickle(data,"preprocessed_words3")
    data = read_pickle("preprocessed_words3")
    data = data.dropna()
    tokens = data["comment"].values.astype('U')
    # tokens = [token.split(" ") for token in tokens]
    # train_word2vec(tokens)
    # print(data.columns)
    tokens = [token.split(" ") for token in tokens]
    # x_tr, count_vect = word_to_vect(data)
    # train_doc2vec(tokens)
    # train_svc(data, read_pickle("w2v_model5"), "SVC")
    # train_model(data, read_pickle("ft_model2"), "SVC", False)
    train_svc_ft(data, read_pickle("ft_model2"))
    # train_word2vec(tokens)
    # train_fasttext(tokens)
    # w2v_model = read_pickle("w2v_model4")
    # ft_model = read_pickle("ft_model1")
    # print(w2v_model)
    # print(ft_model)
    # plt = tsne_scatterplot(w2v_model, "женщина", ["проблема"])
    # plt = tsne_scatterplot(ft_model, "женщина", ["проблема"])
    # plt.show()
    # print(ft_model.wv.most_similar(positive=["женщина"]))
    # print(w2v_model.wv.most_similar(positive=["женщина"]))
    # print(w2v_model.wv.vectors)
    # plt = tsne_scatterplot(w2v_model, "путин", ["байден"])
    # plt.show()
    # print(w2v_model.wv.most_similar(positive=["женщина"]))
    # print(w2v_model.wv.most_similar(positive=["путин"]))
    # X = ft_model.wv[ft_model.wv.vocab]
    # X = w2v_model.wv[w2v_model.wv.vocab]
    # tsne = TSNE(n_components=2)
    # X_tsne = tsne.fit_transform(X)

    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    # plt.show()
    # data_x = []
    # data_y = []
    # for i in range(11200):
    #    try:
    #        data_x.append(float(data["toxic"][i]))
    #    except:
    #        print(i)
    #        try:
    #            print(data["toxic"][i])
    #        except:
    #            print("mistake")
    # write_pickle("preprocessed_words2",data )
    # print(sp.stats.normaltest(data_x))
    # print(sp.stats.normaltest(data_y))
    # print(coeff_corel(data_x, data_y))

    # x_tr, count_vect = word_to_vect(data)
    # tfidf, tfidf_transformer = tf_idf(x_tr)
    # print(x_tr.toarray())
    # print(count_vect.vocabulary_)
    # x_tr.toarray()
    # matrix_freq = np.asarray(x_tr.sum(axis=0)).ravel()
    # s = [np.array(i) for i in x_tr.toarray()]
    # s.append(matrix_freq)
    # s = [np.array(count_vect.get_feature_names())] + s
    # final_matrix = np.array([np.array(count_vect.get_feature_names()), matrix_freq])
    # final_matrix2 = np.array(s)
    # write_pickle(final_matrix, "matrix_count_vect2")
    # final_matrix = read_pickle("matrix_count_vect2")
    # df = pd.DataFrame(final_matrix)
    # df.to_excel("matrix_count_vect2.xlsx", header=False)
    # df = pd.DataFrame(final_matrix2)
    # df.to_excel("matrix_count_vect2.xlsx", header=False)
    # tfidf.toarray()
    # matrix_freq = np.asarray(tfidf.sum(axis=0)).ravel()
    # final_matrix = np.array([np.array(tfidf_transformer.get_params()), matrix_freq])
    # write_pickle(final_matrix, "matrix_tfidf")
    # print(final_matrix)
