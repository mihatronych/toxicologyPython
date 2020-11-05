import sys, os, re, csv, codecs,  pandas as pd
import vk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import nltk
import random

class PreProcessSome:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + list('``'))

    def processText(self, list_of_docs):
        processedDocs = []
        for doc in list_of_docs:
            processedDocs.append([doc[0], self._processText(doc[1]), doc[2], doc[3], doc[4], doc[5], doc[6], doc[7]])
        return processedDocs

    def _processText(self, doc):
        doc = doc.lower()  # convert text to lower-case
        doc = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', doc)  # remove URLs
        doc = re.sub('@[^\s]+', 'AT_USER', doc)  # remove usernames
        doc = re.sub(r'#([^\s]+)', r'\1', doc)  # remove the # in #hashtag
        doc = word_tokenize(doc)  # remove repeated characters (helloooooooo into hello)
        return [word for word in doc if word not in self._stopwords]


def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for docs in preprocessedTrainingData:
        all_words.extend(docs[1])
    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    return word_features


def extract_features(message, wf):
    message_words = set(message)
    features = {}
    for word in wf:
        features['contains(%s)' % word] = (word in message_words)
    return features

if __name__ == '__main__':
    session = vk.Session()
    vk_api = vk.API(session, v="5.92")

    path = 'input/'

    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:1000]
    test = trainx.values[640:650]
    ppr = PreProcessSome()
    ppr = ppr.processText(train)
    testppr = PreProcessSome()
    testppr = testppr.processText(test)
    wf = buildVocabulary(ppr)
    tutu = []
    for i in ppr:
        tutu.append((extract_features(i[1],wf), "toxic" if i[2]==1 else "untoxic"))
    random.shuffle(tutu)
    train_x = tutu[:900]
    test_x = tutu[900:]
    model = nltk.NaiveBayesClassifier.train(train_x)
    model.show_most_informative_features(100)
    acc = nltk.classify.accuracy(model, test_x)
    print("Accuracy:", acc)
    for i in range(len(testppr)):
        t_features = extract_features(testppr[i][1],wf)
        print(test[i], " : ", model.classify(t_features))

