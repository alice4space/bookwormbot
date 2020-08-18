import sys
import pickle
import os
import spacy
import warnings
import csv

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_lg")

botName = sys.argv[1]
filename = sys.argv[2]

print('training on : ' + filename)

doc_counter = 0
texts = ''

files = []
start = [0]
sents = []

with open(filename, mode='r',encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # print(row.keys())
        # print(row["Text"])
        # print(row["\ufeffTitle"])
        doc = row["Text"]
        doc = doc.replace('\n', ' ')
        doc = nlp(doc)
        sents = sents + [sent.string.strip() for sent in doc.sents]
        start.append(len(sents))
        files.append(row["\ufeffTitle"])
        doc_counter += 1
start.pop()



# print(sents)
# print(files)
# print(start)

print("encoding ....")

embds = model.encode(sents)

botPath = 'C:\\Users\\Alice Barthe\\PycharmProjects\\mdrs_bots\\bots\\' + botName + '\\'
os.mkdir(botPath)
pickle.dump(sents, open(botPath + 'sents.p', 'wb'))
pickle.dump(files, open(botPath + 'files.p', 'wb'))
pickle.dump(start, open(botPath + 'start.p', 'wb'))
pickle.dump(embds, open(botPath + 'embds.p', 'wb'))

print('Hello. My name is ' + botName + ' and I trained on ' + str(doc_counter) + ' documents.')