import spacy
import pickle
import warnings
import numpy
import os
import sys
import colorama

warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

botName = sys.argv[1]

botPath = 'C:\\Users\\Alice Barthe\\PycharmProjects\\mdrs_bots\\bots\\' + botName + '\\'
sents = pickle.load(open(botPath + 'sents.p', 'rb'))
files = pickle.load(open(botPath + 'files.p', 'rb'))
start = pickle.load(open(botPath + 'start.p', 'rb'))
embds = pickle.load(open(botPath + 'embds.p', 'rb'))

colorama.init()

# print("trained on :")
# for f in files:
#     head_tail = os.path.split(f)
#     print(head_tail[1])


def cosine_similarity(a, b):
    res = numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
    return res


def find_similar(user_input_fun):
    print('analysing : ' + user_input_fun)
    # user = nlp(user_input_fun)
    # similarity = [user.similarity(nlp(sent)) for sent in sents]
    input_embd = model.encode(user_input_fun)
    input_embd = input_embd[0, :]
    similarity = [cosine_similarity(embd, input_embd) for embd in embds]
    index = numpy.argsort(similarity)
    return index, similarity


def extract_whole(id_doc_fun, id_sen_fun):
    string_whole = ""
    for i in range(start[id_doc_fun], id_sen_fun):
        string_whole = string_whole + " " + sents[i]
    string_whole = string_whole + " " + colorama.Fore.LIGHTYELLOW_EX + sents[id_sen_fun] + colorama.Fore.RESET
    for i in range(id_sen_fun+1, start[id_doc_fun + 1]):
        string_whole = string_whole + " " + sents[i]
    return string_whole


continue_dialogue = True
index_cur = None
similarity_cur = None
id_doc = 0
my_index = 0
answ_cur = 0
index_cur = []

print(colorama.Fore.LIGHTGREEN_EX + "Hello, I am your friend " + botName + ". You can ask me any question :)" + colorama.Fore.RESET)
while continue_dialogue:
    print(colorama.Fore.BLUE + 'tell me' + colorama.Fore.RESET)
    human_text = input()
    if human_text != 'bye':
        if human_text.startswith('ask '):
            answ_cur = 0
            index_cur, similarity_cur = find_similar(human_text[4:])
            my_index = index_cur[-1-answ_cur]
            print(colorama.Fore.LIGHTYELLOW_EX + str(similarity_cur[my_index]) + colorama.Fore.RESET)
            print(colorama.Fore.LIGHTYELLOW_EX + sents[my_index] + colorama.Fore.RESET)
        else:
            if human_text.startswith('next'):
                if len(index_cur) != 0 and answ_cur+1 < len(index_cur):
                    answ_cur += 1
                    my_index = index_cur[-1 - answ_cur]
                    print(colorama.Fore.LIGHTYELLOW_EX + str(similarity_cur[my_index]) + colorama.Fore.RESET)
                    print(colorama.Fore.LIGHTYELLOW_EX + sents[my_index] + colorama.Fore.RESET)
                else:
                    print(colorama.Fore.RED + 'nothing else corresponds' + colorama.Fore.RESET)
                    answ_cur += 1
            else:
                if human_text.startswith('where'):
                    if len(index_cur) != 0:
                        id_doc = 0
                        boolContinue = start[id_doc+1] <= my_index
                        while boolContinue:
                            id_doc += 1
                            if id_doc+1 >= len(start):
                                boolContinue = False
                            else:
                                boolContinue = start[id_doc+1] <= my_index
                        print(colorama.Fore.LIGHTYELLOW_EX + files[id_doc] + colorama.Fore.RESET)
                else:
                    if human_text.startswith('search '):
                        answ_cur = 0
                        user_input = human_text[7:]
                        user_input = user_input.lower()
                        print(colorama.Fore.LIGHTCYAN_EX + 'searching for : ' + user_input + colorama.Fore.RESET)
                        index_cur = [i for i in range(0, len(sents)) if user_input in sents[i]]
                        similarity_cur = [1] * len(sents)
                        if len(index_cur) != 0:
                            my_index = index_cur[-1 - answ_cur]
                            print(colorama.Fore.LIGHTCYAN_EX + "answer number " + str(answ_cur+1) + " out of " + str(len(index_cur)) + colorama.Fore.RESET)
                            print(colorama.Fore.LIGHTYELLOW_EX + sents[my_index] + colorama.Fore.RESET)
                        else:
                            print(colorama.Fore.RED + 'I did not find anything corresponding' + colorama.Fore.RESET)
                    else:
                        if human_text.startswith('whole'):
                            print(colorama.Fore.LIGHTCYAN_EX + "printing the whole document " + files[id_doc] + colorama.Fore.RESET)
                            print(extract_whole(id_doc, my_index))
                        else:
                            print(colorama.Fore.RED + 'I did not get that' + colorama.Fore.RESET)
    else:
        continue_dialogue = False
        print(colorama.Fore.LIGHTGREEN_EX + "Ciao :) " + colorama.Fore.RESET)
