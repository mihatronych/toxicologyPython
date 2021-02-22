import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from nltk.stem.snowball import SnowballStemmer
import pickle
from sklearn.metrics import accuracy_score
import pymorphy2
from collections import defaultdict
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from spacy.lang.ru import Russian
import spacy
import csv
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import fasttext

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
        if row!="":
            data.append(row + ";" + "0")
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

def preprocessing_data(text):
    stop_words = set(stopwords.words('russian') + list(punctuation)) #список стоп-слов
    stemmer = SnowballStemmer("russian")
    morph = pymorphy2.MorphAnalyzer() #для лемматизации
    text = text.lower() #приведение слов к строчным
    text = re.sub(patterns, ' ', text, flags=re.MULTILINE)#удаляем знаки пунктуации
    # text = re.sub('[^а-яА-Я]', ' ', text, flags=re.MULTILINE)#удаляем знаки пунктуации
    tokens = word_tokenize(text) #разделяем слова на отдельные токены
    text = [word for word in tokens if word not in stop_words] #удаляем стоп-слова
    text = [morph.normal_forms(word.strip())[0] for word in text] #производим стемминг
    # text = [stemmer.stem(word) for word in text]  # производим стемминг
    text = ' '.join(text)
    return text


def write_pickle(file, name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(file, f)


def read_pickle(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)

def preprocess(input_data):
    train_db = pd.read_csv(input_data, sep=";", error_bad_lines=False)  # получение датасета
    train_db = train_db[:4000]
    print("Препроцессинг")
    train_db['comment'] = train_db['comment'].map(preprocessing_data) #подготовка данных к обработке
    return train_db

def word_to_vect(data):
    print("Векторизация")
    train_db = data# предобработка
    # разделение на тренировочную и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
                                                        test_size=0.2,
                                                        random_state=42)
    # формирование словаря
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    write_pickle(count_vect, 'count_vect')
    return X_train_counts

def tf_idf(data):
    X_train_counts = data
    # формирование TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    write_pickle(tfidf_transformer, 'tfidf_transformer')
    return X_train_tfidf

def word_freq_func(data):
    word_freq = defaultdict(int)
    for tokens in data.iloc[:]:
        for token in tokens:
            word_freq[token] += 1
    return word_freq


def train_word2vec(tokens):
    w2v_model = Word2Vec(min_count=5, window=2, size=300, negative=10, alpha=0.03, min_alpha=0.0007, sample=6e-5, sg=1)
    w2v_model.build_vocab(tokens)
    print(w2v_model.corpus_count)
    w2v_model.train(tokens, total_examples=w2v_model.corpus_count, epochs=300, report_delay=1)
    w2v_model.init_sims(replace=True)
    write_pickle(w2v_model, 'w2v_model')
    return w2v_model

def tsne_scatterplot(model, word, list_names):
    """Plot in seaborn the results from the t-SNE dimensionality reduction
    algorithm of the vectors of a query word,
    its list of most similar words, and a list of words."""
    vectors_words = [model.wv.word_vec(word)]
    word_labels = [word]
    color_list = ['red']

    close_words = model.wv.most_similar(word)
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
        ).set_size(15)

    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
    plt.title('t-SNE visualization for {}'.format(word.title()))
    return plt

def some_spicy_features_extraction(input_data, nlp): # version pymorphy 0.9
    with open(input_data, "r", encoding="utf-8") as f_csv:
        data = csv_reader(f_csv)
    loc_data = data[1:len(data)]
    loc_data = [i.split(";") for i in loc_data]
    for k in range(len(loc_data)):
        row = loc_data[k]
        res = nlp(row[0])
        for ent in res.ents:
            if len(ent) != 0:
                #print(ent.text, ent.start_char, ent.end_char, ent.label_)
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

if __name__ == '__main__':
    # nlp = spacy.load("ru_core_news_lg")
    messages = ["Верблюдов-то за что? Дебилы, бл...",
                "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. Если бы хохлов не было, кисель их бы придумал.",
                "Какой чудесный день!",
                "ты вообще отстойный, фу таким быть"]
    input_data = 'vk_comments_DS2.csv' #для новой прогонки
    rude_feature_extraction(input_data)
    # some_spicy_features_extraction(input_data, nlp) # функция доставания некоторых признаков:именованные,настроение
    # data = preprocess(input_data)
    # data = data.dropna()
    # print(data)
    # tokens = data["comment"].values.astype('U')
    # x_tr = word_to_vect(data)

    # tokens = [token.split(" ") for token in tokens]
    # train_word2vec(tokens)
    # w2v_model = read_pickle("w2v_model")
    # print(w2v_model)
    # plt = tsne_scatterplot(w2v_model, "женщина", ["проблема"])
    # plt.show()