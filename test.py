import sys, os, re, csv, codecs,  pandas as pd
import vk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import nltk
import random
import pickle

class PreProcessSome:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + list('``'))

    def processText(self, list_of_docs):
        processedDocs = []
        for doc in list_of_docs:
            processedDocs.append([doc[0], self.processDoc(doc[1]), doc[2], doc[3], doc[4], doc[5], doc[6], doc[7]])
        return processedDocs

    def processDoc(self, doc):
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

def pretrain():
    path = 'input/'
    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3000]
    ppr = PreProcessSome()
    ppr = ppr.processText(train)
    wf = buildVocabulary(ppr)
    tutu = []
    for i in ppr:
        tutu.append((extract_features(i[1], wf), "toxic" if i[2] == 1 else "untoxic"))
    random.shuffle(tutu)
    train_x = tutu[:2700]
    model = nltk.NaiveBayesClassifier.train(train_x)
    test_x = tutu[2700:]
    model.show_most_informative_features(20)
    acc = nltk.classify.accuracy(model, test_x)
    path = "output/"
    f = open(f'{path}my_classifier.pickle', 'wb')
    pickle.dump(model, f)
    f.close()
    print("Accuracy:", acc)

def classify_with_modelling(message):
    path = 'input/'
    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3000]
    ppr = PreProcessSome()
    ppr_vocab = ppr.processText(train)
    ppr_message = ppr.processDoc(message)
    wf = buildVocabulary(ppr_vocab)
    path = "output/"
    f = open(f'{path}my_classifier.pickle', 'rb')
    model = pickle.load(f)
    f.close()
    model.show_most_informative_features(20)
    t_features = extract_features(ppr_message, wf)
    r_prob = 0.0
    for i in range(5):
        r_prob += model.prob_classify(t_features).prob("toxic")
    r_prob /= 5
    return str("MESSAGE\r\n"+message + " : " +"\n\rRESULT is\n\r "
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r")

def classify_many_with_modelling(messages):
    path = 'input/'
    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3000]
    ppr = PreProcessSome()
    ppred = ppr.processText(train)
    ppr_messages = []
    for message in messages:
        ppr_messages.append(ppr.processDoc(message))
    wf = buildVocabulary(ppred)
    path = "output/"
    f = open(f'{path}my_classifier.pickle', 'rb')
    model = pickle.load(f)
    f.close()
    result = []
    for k in range(len(messages)):
        t_features = extract_features(ppr_messages[k], wf)
        r_prob = 0.0
        for i in range(5):
            r_prob += model.prob_classify(t_features).prob("toxic")
        r_prob /= 5

        result.append(("MESSAGE\r\n"+messages[k] + " : " +"\n\rRESULT is\n\r "
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r"))
    return result

if __name__ == '__main__':
    #session = vk.Session()
    #vk_api = vk.API(session, v="5.92")

    path = 'input/'

    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    #pretrain()
    test = trainx.values[3150:3200]
    messages = []
    for message in test:
        messages.append(message[1])
    [print(i) for i in classify_many_with_modelling(messages)]
    #print(messages[0])
    #print(classify_with_modelling(messages[0]))

