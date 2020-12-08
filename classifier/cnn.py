import os

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
from sklearn.svm import SVC
from nltk.stem.snowball import SnowballStemmer
import pickle
from sklearn.metrics import accuracy_score
#from pymystem3 import Mystem
import pymorphy2 as pm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

def preprocessing_data(text):
    stop_words = set(stopwords.words('russian') + list(punctuation)) #список стоп-слов
    stemmer = SnowballStemmer("russian")
    #morph = pymorphy2.MorphAnalyzer() #для лемматизации
    text = text.lower() #приведение слов к строчным
    text = re.sub('[^а-яА-Я]', ' ', text, flags=re.MULTILINE)#удаляем знаки пунктуации
    tokens = word_tokenize(text) #разделяем слова на отдельные токены
    text = [word for word in tokens if word not in stop_words] #удаляем стоп-слова
    text = [stemmer.stem(word) for word in text] #производим стемминг
    text = ' '.join(text)
    return text

def write_pickle(file, name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(file, f)


def read_pickle(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)

def save_clear_data(input_data):
    train_db = pd.read_csv(input_data)
    train_db['comment'] = train_db['comment'].map(preprocessing_data)
    columns = ['comment', 'toxic']
    df = pd.DataFrame(train_db, columns=columns)
    df.to_csv(r'clear_data.csv', mode='a', header=True, index=False)

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()

    # слои свертки
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2)
    self.conv1_s = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=2, padding=2)
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
    self.conv2_s = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1)
    self.conv3_s = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1)

    self.flatten = nn.Flatten()
    # гордый полносвязный слой
    self.fc1 = nn.Linear(10, 10)

  # функция для "пропускания" данных через модельку
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv1_s(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv2_s(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv3_s(x))

    x = self.flatten(x)
    x = self.fc1(x)
    x = F.softmax(x)

    return x

def training_data(input_data):
    batch_size = 64  # размер батча
    learning_rate = 1e-3  # шаг оптимизатора
    epochs = 200  # сколько эпох обучаемся
    simple_cnn = SimpleCNN()  # перемещаем на gpu, чтобы училось быстрее, жилось веселее
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(simple_cnn.parameters(), lr=learning_rate)
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

    x_train_tensor = torch.tensor(X_train.astype(np.float32))
    x_test_tensor = torch.tensor(X_test.astype(np.float32))

    y_train_tensor = torch.tensor(y_train.astype(np.int))
    y_test_tensor = torch.tensor(y_test.astype(np.int))

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for epoch in range(epochs):  # итерируемся 200 эпох
        simple_cnn.train()
        train_samples_count = 0  # общее количество картинок, которые мы прогнали через модельку
        true_train_samples_count = 0  # количество верно предсказанных картинок
        running_loss = 0

        for batch in train_loader:
            x_data = batch[0]#.cuda()  # данные тоже необходимо перемещать на gpu
            y_data = batch[1]#.cuda()

            y_pred = simple_cnn(x_data)
            loss = criterion(y_pred, y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # обратное распространение ошибки

            running_loss += loss.item()

            y_pred = y_pred.argmax(dim=1, keepdim=False)
            true_classified = (y_pred == y_data).sum().item()  # количество верно предсказанных картинок в текущем батче
            true_train_samples_count += true_classified
            train_samples_count += len(x_data)  # размер текущего батча

        train_accuracy = true_train_samples_count / train_samples_count
        print(f"[{epoch}] train loss: {running_loss}, accuracy: {round(train_accuracy, 4)}")  # выводим логи

        # прогоняем тестовую выборку
        simple_cnn.eval()
        test_samples_count = 0
        true_test_samples_count = 0
        running_loss = 0

        for batch in test_loader:
            x_data = batch[0]#.cuda()
            y_data = batch[1]#.cuda()

            y_pred = simple_cnn(x_data)
            loss = criterion(y_pred, y_data)

            loss.backward()

            running_loss += loss.item()

            y_pred = y_pred.argmax(dim=1, keepdim=False)
            true_classified = (y_pred == y_data).sum().item()
            true_test_samples_count += true_classified
            test_samples_count += len(x_data)

        test_accuracy = true_test_samples_count / test_samples_count
        print(f"[{epoch}] test loss: {running_loss}, accuracy: {round(test_accuracy, 4)}")

if __name__ == '__main__':
    input_data = 'labeled_ru_ds.csv'  # для новой прогонки
    input_data = 'clear_data.csv'  # лемматизированный датасет
    training_data(input_data)

    messages = ["Верблюдов-то за что? Дебилы, бл...",
                "Хохлы, это отдушина затюканого россиянина, мол, вон, а у хохлов еще хуже. Если бы хохлов не было, кисель их бы придумал.",
                "Какой чудесный день!",
                "ты вообще отстойный, фу таким быть"]