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

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

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

def tf_idf(data): # пока не работает
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

if __name__ == '__main__':
    messages = ["Верблюдов-то за что? Дебилы, бл...",
                "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. Если бы хохлов не было, кисель их бы придумал.",
                "Какой чудесный день!",
                "ты вообще отстойный, фу таким быть"]
    input_data = 'vk_comments_DS.csv' #для новой прогонки
    data = preprocess(input_data)
    data = data.dropna()
    print(data)
    tokens = data["comment"].values.astype('U')
    x_tr = word_to_vect(data)
    print(x_tr)
    tokens = [token.split(" ") for token in tokens]
    train_word2vec(tokens)
    w2v_model = read_pickle("w2v_model")
    print(w2v_model)
    plt = tsne_scatterplot(w2v_model, "женщина", ["проблема"])
    plt.show()