"""
for extracting and combining features
12/9/2020
B.Wagner & J. Chanenson
"""

import math
from infuse_data import load_posneg
import sys
from nltk import sent_tokenize, word_tokenize, pos_tag
import json

def test_features(men_data, women_data):
    """
    Quick function that tests the features
    @param
        - men_data: list of male data each element in list is [text,tag]
        - women_data: list of female data each element in list is [text,tag]
    @return None
    """
    for i in range(10):
        m = find_features(men_data[i][0],'M')
        w = find_features(women_data[i][0], 'F')

def find_features(text, gender, args):
    """
    The main workhorse of this file
    @params
        -text: a string representing one post's text
        -gender: 'M' or 'F'
        -args: args from the parser in genderc.py
    @return
        list of feature values
    """

    tokens = word_tokenize(text)
    w = len(tokens)
    #print('w', w)

    excl = text.count('!')/w
    commas = text.count(',')/w
    periods = text.count('.')/w
    questions = text.count('?')/w
    pos, neg = posNeg(text)
    emotion = pos + neg
    lexical = lexical_diversity(tokens)

    #print("features %d %d %d %d" %(excl, periods, commas, questions))
    weights = {}
    #smoothing, formatting as log
    # these are blog weights
    if args.blog:
        mweights = {'pos': 1.403100307559178,
                    'neg': 0.7200101004754397,
                    'emotion': 1.0643320994920809,
                    'excl': 0.3798232943492426,
                    'commas': 2.8321198141988,
                    'periods': 2.4318613852465534,
                    'questions': 2.287983529574044,
                    'lexical': 1.138903830364552}

        fweights = {'pos': 2.9975080100635387,
                    'neg': 2.028113734147325,
                    'emotion': 1.5027740125534077,
                    'excl': 2.9173182208269117,
                    'commas': 0.6342309720319979,
                    'periods': 0.12994407120358992,
                    'questions': 1.7368091880070426,
                    'lexical': 2.9503370065915075}


    if args.NYT:
        mweights = {'pos': 2.1149271380290253,
                    'neg': 0.4326952563350226,
                    'emotion': 1.5565790189584243,
                    'excl': 0.10432536220285579,
                    'commas': 2.10734105017255,
                    'periods': 2.874545859228758,
                    'questions': 2.8370598975005015,
                    'lexical': 1.0127643591330733}

        fweights = {'pos': 2.895645668441445,
                    'neg': 2.9347833138100947,
                    'emotion': 0.17893399748499558,
                    'excl': 2.457673331114158,
                    'commas': 0.36453956217862815,
                    'periods': 2.294893476098146,
                    'questions': 0.975527409608784,
                    'lexical': 0.9965078692737547}

    if gender == 'M':
        weights = mweights
    else:
        weights = fweights

    features = {
        "pos":pos,
        "neg":neg,
        "emotion":emotion,
        "excl":excl,
        "commas":commas,
        "periods":periods,
        "questions":questions,
        "lexical":lexical
    }

    #BIG IMPORTANT BAYESIAN EQUATION
    agg = 1
    for key,val in features.items():
        val = val*weights[key]
        val += 1e-5
        agg *= val
        if val == 0.0:
            print('\n\nZERO', key, val)
            sys.exit()
        features[key] = val

    return agg

def tune_features(text, gender, mw, ww):
    """
    This function is used to tune features.
    Helper function for tune_classifier.py
    @params
        -text: a string representing one post's text
        -gender: 'M' or 'F'
        - mw: men weights (dict)
        - ww: women weights (dict)
    @return - list of feature values
    """

    tokens = word_tokenize(text)
    w = len(tokens)
    #print('w', w)

    excl = text.count('!')/w
    commas = text.count(',')/w
    periods = text.count('.')/w
    questions = text.count('?')/w
    pos, neg = posNeg(text)
    emotion = pos + neg
    lexical = lexical_diversity(tokens)

    #print("features %d %d %d %d" %(excl, periods, commas, questions))
    weights = {}
    #smoothing, formatting as log
    mweights = mw
    fweights = ww

    if gender == 'M':
        weights = mweights
    elif gender == 'F':
        weights = fweights
    else:
        weights = mweights

    features = {
        "pos":pos,
        "neg":neg,
        "emotion":emotion,
        "excl":excl,
        "commas":commas,
        "periods":periods,
        "questions":questions,
        "lexical":lexical
    }

    #BIG IMPORTANT BAYESIAN EQUATION
    agg = 1
    for key,val in features.items():
        val = val*weights[key]
        val += 1e-5
        agg *= val
        if val == 0.0:
            print('\n\nZERO', key, val)
            sys.exit()
        features[key] = val

    return agg

def load_tokens(gender_data):
    """
    Given gender data, generate a list of all tokens.
    @params - gender_data: list of gender data with tag each item in list looks
                like [text, tag]
    @return list of tokens
    """

    tokens = []
    for item in gender_data:
        tokens.extend(word_tokenize(item[0]))

    return tokens

def POS_time(data):
    """
    A function that generates POS tags for running text
    @param - data: running text
    @return posLst: a 3D array
                        - posLst[0] = an item from the corpus
                        - posLst[0][0] = a word, tag tuple
    """
    #saved all pos results as .json :)

    """
    posLst = POS_time(men_data)
    outputFile = open("tagged_data_male.json", 'w')
    json.dump(posLst,outputFile)
    outputFile.close()

    posLst = POS_time(women_data)
    outputFile = open("tagged_data_female.json", 'w')
    json.dump(posLst,outputFile)
    outputFile.close()
    """
    print("In pos time")
    posLst = []
    entry = []
    # split running text on sentances, create list of lists where each blog
    #post has one list of sentences
    chunked = [sent_tokenize(post[0]) for post in data]

    #tokenize each sentance, then POS Tag each tokenzied sentance
    for post in chunked:
        for sentence in post:
            entry.extend(pos_tag(word_tokenize(sentence))) #glue each line into entry
        posLst.append(entry) #append entry back to corpus
        entry = []

    return posLst

def posNeg(gender_data):
    """
    Calculates based of pos neg words.
    @params  - gender_data: running text
    @return - dict of tokens to probabilities based on data
    """

    dict =  {}
    #accumulate frequencies of each word, create dict of frequencies
    tokens = word_tokenize(gender_data)
    for token in tokens:
        dict[token] = dict.setdefault(token,0) + 1
    dict["<UNK>"] = 0

    v = len(dict) #number of unique words
    w = sum(dict.values()) #number of tokens

    #pos neg
    posDict, negDict = load_posneg()

    # use intersection to find the pos and neg words that are in corpus
    posIntersect = posDict.keys() & dict.keys()
    negIntersect = dict.keys() & negDict.keys()

    numPos = len(posIntersect)
    numNeg = len(negIntersect)
    oPS = numPos

    numStrongWords = numPos + numNeg

    return numPos/w, numNeg/w #normalized

def passivePOS(dict, gender_data, smoothing):
    """
    Calculates based of occurance of passive voice.
    params  - dict: the dict of tokens
            - gender_data: running text
            - smoothing: laplace smoothing variable
    return - dict of tokens to probabilities based on data
    """

    v = len(dict)
    w = sum(dict.values())

    #POS Tagging
    POS_time(gender_data)

    for word, freq in dict.items():
        prob = ((freq + smoothing)/(w+smoothing*v))
        dict[word] = math.log(prob)

    return dict

def counter(gender_text, query):
    """
    params
        - gender_data: running text
        - query: string character to count in text
    return - count of query
    """

    count = 0

    for character in gender_text:
        if character == query:
            count += 1

    return count

def lexical_diversity(text):
    """
    Feature that calculates lexical_diversity
    @params:
        -text: list of tokens
    @returns: float score
    """
    return len(set(text))/len(text)

def main():
    print("I contain feature related functions for genderc.py and \
    tune_classifier.py")

if __name__ == '__main__':
    main()
