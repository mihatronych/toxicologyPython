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
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import SnowballStemmer
import pickle
import pymorphy2 as pm
from sklearn.model_selection import GridSearchCV

def preprocessing_data(text):
    stop_words = set(stopwords.words('russian') + list(punctuation)) #список стоп-слов
    stemmer = SnowballStemmer("russian")
    # m = Mystem() #для лемматизации
    morph = pm.MorphAnalyzer()
    text = text.lower() #приведение слов к строчным
    text = re.sub('[^а-яА-Я]', ' ', text, flags=re.MULTILINE)#удаляем знаки пунктуации
    tokens = word_tokenize(text) #разделяем слова на отдельные токены
    text = [word for word in tokens if word not in stop_words] #удаляем стоп-слова
    #text = [stemmer.stem(word) for word in text] #производим стемминг
    text = [morph.parse(word)[0].normal_form for word in text] #производим лемматизацию
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


def training_data(input_data):
    train_db = pd.read_csv(input_data)  # получение датасета
    # train_db = train_db[:100]
    #print("Препроцессинг")
    #train_db['comment'] = train_db['comment'].map(preprocessing_data)  # подготовка данных к обработке
    # разделение на тренировочную и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
                                                        test_size=0.33,
                                                        random_state=42)

    print("Векторизация")
    # формирование словаря
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_counts.shape
    write_pickle(count_vect, 'count_vect')

    print("Векторизация TF-IDF")
    # формирование TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    write_pickle(tfidf_transformer, 'tfidf_transformer')

    print("Обучение MLP классификатора")
    # тренировка классификтора
    #model = MLPClassifier(solver="sgd", activation="relu", learning_rate_init=0.001, max_iter=500, alpha=1e-5, hidden_layer_sizes=(100,100,100), verbose=10, tol=0.000000001, random_state=21).fit(X_train_tfidf, y_train)
    model = MLPClassifier(max_iter=40).fit(X_train_tfidf, y_train) #0.8738435660218671
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train_tfidf, y_train)

    write_pickle(model, 'modelMLP')

    # формирование тренировочной выборки
    count_vect2 = read_pickle('count_vect')
    X_new_counts = count_vect2.transform(X_test)
    tfidf_transformer2 = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer2.transform(X_new_counts)

    print("Оценка точности MLP классификатора")
    # оценка точности классификатора
    predicted = clf.predict(X_new_tfidf)
    acc = np.mean(predicted == y_test)
    print("Точность: ", acc)  # 0.86565...
    # 0.866484440706476 если юзать pymorphy2


def classifier(messages):
    clear_message = map(preprocessing_data, messages)
    count_vect = read_pickle('count_vect')
    X_new_counts = count_vect.transform(clear_message)
    tfidf_transformer = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    model = read_pickle('modelMLP')
    predicted = model.predict_proba(X_new_tfidf)
    return zip(messages, predicted)


def save_clear_data(input_data):
    train_db = pd.read_csv(input_data)
    train_db['comment'] = train_db['comment'].map(preprocessing_data)
    columns = ['comment', 'toxic']
    df = pd.DataFrame(train_db, columns=columns)
    df.to_csv(r'clear_data.csv', mode='a', header=True, index=False)


if __name__ == '__main__':
    input_data = 'labeled_ru_ds.csv' #для новой прогонки
    input_data = 'clear_data.csv' #лемматизированный датасет
    training_data(input_data)

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