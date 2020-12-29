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
from sklearn import model_selection
from sklearn.svm import SVC
from nltk.stem.snowball import SnowballStemmer
import pickle
from sklearn.metrics import accuracy_score
#from pymystem3 import Mystem
import pymorphy2 as pm
from sklearn.ensemble import BaggingClassifier

def preprocessing_data(text):
    stop_words = set(stopwords.words('russian') + list(punctuation)) #список стоп-слов
    stemmer = SnowballStemmer("russian")
    # m = Mystem() #для лемматизации
    morph = pm.MorphAnalyzer()
    text = text.lower() #приведение слов к строчным
    text = re.sub('[^а-яА-Я]', ' ', text, flags=re.MULTILINE)#удаляем знаки пунктуации
    tokens = word_tokenize(text) #разделяем слова на отдельные токены
    text = [word for word in tokens if word not in stop_words] #удаляем стоп-слова
    text = [stemmer.stem(word) for word in text] #производим стемминг
    #text = [morph.parse(word)[0].normal_form for word in text] #производим лемматизацию
    # text = ' '.join(text)
    # text = m.lemmatize(text)
    text = ' '.join(text)
    return text


def write_pickle(file, name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(file, f)


def read_pickle(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)


def training_data(input_data, n_trees, seed):
    train_db = pd.read_csv(input_data)  # получение датасета
    # train_db = train_db[:100]
    # print("Препроцессинг")
    # train_db['comment'] = train_db['comment'].map(preprocessing_data)  # подготовка данных к обработке
    # разделение на тренировочную и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
                                                        test_size=0.33)
    print("Векторизация")
    #формирование словаря
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_counts.shape
    write_pickle(count_vect, 'count_vect')

    print("Векторизация TF-IDF")
    #формирование TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    write_pickle(tfidf_transformer, 'tfidf_transformer')

    print("Обучение BAG классификатора")
    #тренировка классификтора
    #model = BernoulliNB().fit(X_train_tfidf, y_train)
    #write_pickle(model, 'modelBernoulliNB')
    #model = BaggingClassifier(base_estimator=model, n_estimators=n_trees, random_state=seed)


    #формирование тренировочной выборки
    count_vect2 = read_pickle('count_vect')
    X_new_counts = count_vect2.transform(X_test)
    tfidf_transformer2 = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer2.transform(X_new_counts)

    kfold = model_selection.KFold(n_splits=100, random_state=seed, shuffle=True)
    #cart = BernoulliNB()
    cart = SVC(kernel='linear',probability=True)
    model = BaggingClassifier(base_estimator=cart, n_estimators=n_trees, random_state=seed).fit(X_train_tfidf, y_train)
    results = model_selection.cross_val_score(model, X_train_tfidf, y_train, cv=kfold)
    print(results.mean())  # 0.8891111111111111 BernoulliNB
    write_pickle(model, 'modelSVC_bag')
    print("Оценка точности SVC классификатора")
    #оценка точности классификатора
    predicted = model.predict(X_new_tfidf)
    acc = np.mean(predicted == y_test)
    print("Точность: ", acc) #0.8717409587888982
    #0.8732127838519764 если юзать pymorphy2


def classifier(messages):
    clear_message = map(preprocessing_data, messages)
    count_vect = read_pickle('count_vect')
    X_new_counts = count_vect.transform(clear_message)
    tfidf_transformer = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    model = read_pickle('modelSVC_bag')
    predicted = model.predict_proba(X_new_tfidf)
    return zip(messages, predicted)

def save_clear_data(input_data):
    train_db = pd.read_csv(input_data)
    train_db['comment'] = train_db['comment'].map(preprocessing_data)
    columns = ['comment', 'toxic']
    df = pd.DataFrame(train_db, columns=columns)
    df.to_csv(r'clear_data.csv', mode='a', header=True, index=False)

if __name__ == '__main__':
    input_data = 'labeled_ru_ds.csv'  # для новой прогонки
    input_data = 'clear_data.csv'  # лемматизированный датасет
    training_data(input_data, 10, 42)

    messages = ["Верблюдов-то за что? Дебилы, бл...",
                "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. Если бы хохлов не было, кисель их бы придумал.",
                "Какой чудесный день!",
                "ты вообще отстойный, фу таким быть"]

    labeled_messages = classifier(messages)

    counter = 0
    for comment, toxic in labeled_messages:
        print('%r => %s' % (comment, toxic))
        if (counter == 10):
            break
        counter += 1