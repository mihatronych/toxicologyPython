import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import xgboost as xgb
from nltk.stem.snowball import SnowballStemmer
import pickle
import pymorphy2 as pm

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
    with open(r'C:\Users\theme\PycharmProjects\toxicologyPython\classifier\count_vect.pkl', 'rb') as f:
        return pickle.load(f)


def training_data(input_data):
    train_db = pd.read_csv(input_data)  # получение датасета
    # train_db = train_db[:100]
    # print("Препроцессинг")
    # train_db['comment'] = train_db['comment'].map(preprocessing_data)  # подготовка данных к обработке
    # разделение на тренировочную и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(train_db['comment'].values.astype('U'), train_db['toxic'],
                                                        test_size=0.33,
                                                        random_state=42)
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

    # формирование тренировочной выборки
    count_vect2 = read_pickle('count_vect')
    X_new_counts = count_vect2.transform(X_test)
    tfidf_transformer2 = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer2.transform(X_new_counts)

    print("Обучение XGB классификатора")
    #тренировка классификтора
    model = xgb.XGBClassifier(random_state=42, base_score=0.5, booster='dart', colsample_bylevel=1, colsample_bytree=1, gamma=0
                              , learning_rate=0.1, max_delta_step=0, max_depth=20, min_child_weight=1, missing=None,
                              n_estimators=100, n_jobs=6, objective='binary:logistic', reg_alpha=0.1, reg_lambda=1,
                              scale_pos_weight=1, subsample=1, eval_metric="logloss").fit(X_train_tfidf, y_train, early_stopping_rounds=10, eval_set=[(X_new_tfidf, y_test)])
    write_pickle(model, 'modelXGB')



    count_vect2 = read_pickle('count_vect')
    X_new_counts = count_vect2.transform(X_test)
    tfidf_transformer2 = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer2.transform(X_new_counts)

    print("Оценка точности XGB классификатора")
    #оценка точности классификатора
    predicted = model.predict(X_new_tfidf)
    acc = np.mean(predicted == y_test)
    print("Точность: ", acc)


def classifier(messages):
    clear_message = map(preprocessing_data, messages)
    count_vect = read_pickle('count_vect')
    X_new_counts = count_vect.transform(clear_message)
    tfidf_transformer = read_pickle('tfidf_transformer')
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    model = read_pickle('modelXGB')
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