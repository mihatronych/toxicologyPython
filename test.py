import sys, os, re, csv, codecs,  pandas as pd
import vk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import pickle
import zipfile
from nltk.stem.snowball import SnowballStemmer

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


class PreProcessSomeRusnya:
    def __init__(self):
        self._stopwords = set(stopwords.words('russian') + list(punctuation) + list('``'))

    def processText(self, list_of_docs):
        processedDocs = []
        id = 0
        for doc in list_of_docs:
            processedDocs.append([id, self.processDoc(doc[0]), doc[1]])
            id += 1
        return processedDocs

    def processTextOnBigram(self, list_of_docs):
        processedDocs = []
        id = 0
        for doc in list_of_docs:
            processedDocs.append([id, self.bigramProcessDoc(doc[0]), doc[1]])
            id += 1
        return processedDocs

    def processDoc(self, doc):
        stemmer = SnowballStemmer("russian")
        doc = doc.lower()  # convert text to lower-case
        doc = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', doc)  # remove URLs
        doc = re.sub('@[^\s]+', 'AT_USER', doc)  # remove usernames
        doc = re.sub(r'#([^\s]+)', r'\1', doc)  # remove the # in #hashtag
        doc = word_tokenize(doc)  # remove repeated characters (helloooooooo into hello)
        return [stemmer.stem(word) for word in doc if word not in self._stopwords]

    def bigramProcessDoc(self, doc):
        stemmer = SnowballStemmer("russian")
        doc = doc.lower()  # convert text to lower-case
        doc = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', doc)  # remove URLs
        doc = re.sub('@[^\s]+', 'AT_USER', doc)  # remove usernames
        doc = re.sub(r'#([^\s]+)', r'\1', doc)  # remove the # in #hashtag
        doc = word_tokenize(doc)  # remove repeated characters (helloooooooo into hello)
        return [stemmer.stem(doc[i]) + " " + stemmer.stem(doc[i + 1]) for i in range(len(doc) - 1)
                if stemmer.stem(doc[i]) not in self._stopwords]

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
        features["contains "+word] = (word in message_words)
        # features['contains(%s)' % word] = (word in message_words)
    return features

def pretrainNB():
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

def pretrainSVCeng():
    path = 'input/'
    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:2000]
    ppr = PreProcessSome()
    ppr = ppr.processText(train)
    wf = buildVocabulary(ppr)
    tutu = []
    for i in ppr:
        tutu.append((extract_features(i[1], wf), "toxic" if i[2] == 1 else "untoxic"))
    random.shuffle(tutu)
    train_x = tutu[:1800]
    SVC_classifier = SklearnClassifier(SVC(kernel='linear',probability=True))
    SVC_classifier.train(train_x)
    test_x = tutu[1800:]
    acc = nltk.classify.accuracy(SVC_classifier, test_x)
    path = "output/"
    f = open(f'{path}my_svc_classifier.pickle', 'wb')
    pickle.dump(SVC_classifier, f)
    f.close()
    with zipfile.ZipFile(f'{path}svc.zip', mode='w',compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(f'{path}my_svc_classifier.pickle', arcname="my_svc_classifier.pickle")
    os.remove(f'{path}my_svc_classifier.pickle')
    print("Accuracy:", acc)


def pretrainSVCru():
    path = 'input/'
    train_data_file = f'{path}toxic_labeled_comments.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3500]
    ppr = PreProcessSomeRusnya()
    ppr = ppr.processText(train)
    wf = buildVocabulary(ppr)
    tutu = []
    for i in ppr:
        tutu.append((extract_features(i[1], wf), "toxic" if i[2] == 1 else "untoxic"))
    random.shuffle(tutu)
    train_x = tutu[:2750]
    SVC_classifier = SklearnClassifier(SVC(kernel='linear',probability=True))
    SVC_classifier.train(train_x)
    test_x = tutu[2750:]
    acc = nltk.classify.accuracy(SVC_classifier, test_x)
    path = "output/"
    f = open(f'{path}my_ru_svc_classifier.pickle', 'wb')
    pickle.dump(SVC_classifier, f)
    f.close()
    with zipfile.ZipFile(f'{path}svc_ru.zip', mode='w',compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(f'{path}my_ru_svc_classifier.pickle', arcname="my_ru_svc_classifier.pickle")
    os.remove(f'{path}my_ru_svc_classifier.pickle')
    print("Accuracy:", acc)

def pretrainSVCruBigram():
    path = 'input/'
    train_data_file = f'{path}toxic_labeled_comments.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:2000]
    ppr = PreProcessSomeRusnya()
    ppr = ppr.processTextOnBigram(train)
    wf = buildVocabulary(ppr)
    tutu = []
    for i in ppr:
        tutu.append((extract_features(i[1], wf), "toxic" if i[2] == 1 else "untoxic"))
    random.shuffle(tutu)
    train_x = tutu[:1500]
    SVC_classifier = SklearnClassifier(SVC(kernel='linear',probability=True))
    SVC_classifier.train(train_x)
    test_x = tutu[1500:]
    acc = nltk.classify.accuracy(SVC_classifier, test_x)
    path = "output/"
    f = open(f'{path}my_bigram_ru_svc_classifier.pickle', 'wb')
    pickle.dump(SVC_classifier, f)
    f.close()
    with zipfile.ZipFile(f'{path}svc_ru_bigram.zip', mode='w',compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(f'{path}my_bigram_ru_svc_classifier.pickle', arcname="my_bigram_ru_svc_classifier.pickle")
    os.remove(f'{path}my_bigram_ru_svc_classifier.pickle')
    print("Accuracy:", acc)

def classify_with_modelling_NB(message):
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
    return str("MESSAGE\r\n"+message + " : " +"\n\rNB classification RESULT is\n\r "
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r")

# Сделать надо будет потом в одну функцию с ру версией
def classify_with_modelling_SVCeng(message):
    path = 'input/'
    train_data_file = f'{path}train.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3000]
    ppr = PreProcessSome()
    ppr_vocab = ppr.processText(train)
    ppr_message = ppr.processDoc(message)
    wf = buildVocabulary(ppr_vocab)
    path = "output/"
    archive = zipfile.ZipFile(f'{path}svc.zip', 'r')
    f = archive.open("my_svc_classifier.pickle")
    model = pickle.load(f)
    t_features = extract_features(ppr_message, wf)
    r_prob = 0.0
    for i in range(5):
        r_prob += model.prob_classify(t_features).prob("toxic")
    r_prob /= 5
    return str("MESSAGE\r\n"+message + " : " +"\n\rSVC classification RESULT is\n\r "
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r")

def classify_with_modelling_SVCru(message):
    path = 'input/'
    train_data_file = f'{path}toxic_labeled_comments.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3000]
    ppr = PreProcessSomeRusnya()
    ppr_vocab = ppr.processText(train)
    ppr_message = ppr.processDoc(message)
    wf = buildVocabulary(ppr_vocab)
    path = "output/"
    archive = zipfile.ZipFile(f'{path}svc_ru.zip', 'r')
    f = archive.open("my_ru_svc_classifier.pickle")
    model = pickle.load(f)
    t_features = extract_features(ppr_message, wf)
    r_prob = 0.0
    for i in range(5):
        r_prob += model.prob_classify(t_features).prob("toxic")
    r_prob /= 5
    return str("MESSAGE\r\n"+message + " : " +"\n\rSVC classification RESULT is\n\r "
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r")

def classify_many_with_modelling_NB(messages):
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

        result.append(("MESSAGE\r\n"+messages[k] + " : " +"\n\rNB classification RESULT is\n\r "
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r"))
    return result

def classify_many_with_modelling_SVCeng(messages):
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
    archive = zipfile.ZipFile(f'{path}svc.zip', 'r')
    f = archive.open("my_svc_classifier.pickle")
    model = pickle.load(f)
    result = []
    for k in range(len(messages)):
        t_features = extract_features(ppr_messages[k], wf)
        r_prob = 0.0
        for i in range(5):
            r_prob += model.prob_classify(t_features).prob("toxic")
        r_prob /= 5

        result.append(("MESSAGE\r\n"+messages[k] + " : " +"\n\rSVC classification RESULT is\n\r ")
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r")
    return result

def classify_many_with_modelling_SVCru(messages, bi=False):
    path = 'input/'
    train_data_file = f'{path}toxic_labeled_comments.csv'
    trainx = pd.read_csv(train_data_file)
    train = trainx.values[:3000]
    ppr = PreProcessSomeRusnya()
    ppred = ppr.processText(train)
    ppr_messages = []
    for message in messages:
        ppr_messages.append(ppr.processDoc(message))
    wf = buildVocabulary(ppred)
    path = "output/"
    if(bi != False):
        archive = zipfile.ZipFile(f'{path}svc_ru.zip', 'r')
        f = archive.open("my_ru_svc_classifier.pickle")
    else:
        archive = zipfile.ZipFile(f'{path}svc_ru_bigram.zip', 'r')
        f = archive.open("my_bigram_ru_svc_classifier.pickle")
    model = pickle.load(f)
    result = []
    for k in range(len(messages)):
        t_features = extract_features(ppr_messages[k], wf)
        r_prob = 0.0
        for i in range(5):
            r_prob += model.prob_classify(t_features).prob("toxic")
        r_prob /= 5

        result.append(("MESSAGE\r\n"+messages[k] + " : " +"\n\rSVC classification RESULT is\n\r ")
                       + str(r_prob) + " Toxic\n\r " +
                       str(1 - r_prob) + " Untoxic\n\r")
    return result

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
            data.append(row + "#" + "0")
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

if __name__ == '__main__':
    #session = vk.Session()
    #vk_api = vk.API(session, v="5.92")

    path = 'input/'

    #train_data_file = f'{path}train.csv'
    train_data_file = f'{path}toxic_labeled_comments.csv'
    trainx = pd.read_csv(train_data_file)
    #pretrainNB()
    #pretrainSVCeng() #90% точность почти
    pretrainSVCru() #кое-как 70% точность, це фигово, но в принципе угадывает
    #pretrainSVCruBigram() #0.576 отвратительно!
    test = trainx.values[3150:3200]
    messages = []
    for message in test:
        messages.append(message[0])

    #    messages.append(message[1])
    # [print(i) for i in classify_many_with_modelling_NB(messages)]
    # [print(i) for i in classify_many_with_modelling_SVCeng(messages)] # для английской версии
    [print(i) for i in classify_many_with_modelling_SVCru(messages, True)] # для русской версии
    #print(messages[0])
    #print(classify_with_modelling_NB(messages[0]))
    #print(classify_with_modelling_SVCeng(messages[0]))

    #csv_path = "vk_parser/vk_comments.csv"
    #data = []
    #with open(csv_path, "r", encoding='utf-8') as f_obj:
    #    data = csv_reader(f_obj)
    #    data[0] = "comment#toxic"
    #    data = [el.split("#") for el in data]

    #my_list = []
    #fieldnames = data[0]
    #for values in data[1:]:
    #    inner_dict = dict(zip(fieldnames, values))
    #    my_list.append(inner_dict)

    #path = "vk_parser/vk_comments_DS.csv"
    #csv_dict_writer(path, fieldnames, my_list)

