"""
This code is adapted from Adriana and Bayliss's Lab5
It performs naive bayes on gender data
11/7/2020
J. Chanenson & B. Wagner
"""

import os
import math
from operator import itemgetter
import argparse
from nltk import sent_tokenize, word_tokenize, pos_tag
"""
import nltk
>>> nltk.download('punkt')
[nltk_data] Error loading punkt: <urlopen error [SSL:
[nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed:
[nltk_data]     unable to get local issuer certificate (_ssl.c:1123)>
False
>>> import ssl
>>> try:
...     _create_unverified_https_context = ssl._create_unverified_context
... except AttributeError:
...     # Legacy Python that doesn't verify HTTPS certificates by default
...     pass
... else:
...     # Handle target environment that doesn't support HTTPS verification
...     ssl._create_default_https_context = _create_unverified_https_context
...
>>>
>>> nltk.download('punkt')
"""
import json
import random
from datetime import datetime

# things we wrote
from features import *
from infuse_data import read_NYT, read_blogs, load_posneg, balanced_split


class GenderFilter(object):
    def __init__(self, men_data, women_data, smoothing, classifier_type):
        """
        Init female filter object.
        params - men_data: female training set, list of ['string', 'gender'] lists
                 male_dir: male training set, same format as above
                 smoothing: laplace smoothing parameter for calculating probs
        fiels  - self.female_prob: probability of female category
                 self.male_prob: probability of male category
                 self.female_dict: dict of probabilities of words occuring in female
                 self.male_dict: dict of probabilities of words occuring in male
                 self.intersection: list of common keys between female/male dicts
        """
        self.classifier_type = classifier_type

        all = len(men_data) + len(women_data)
        self.men_tokens = load_tokens(men_data)
        self.women_tokens = load_tokens(women_data)

        self.male_prob = math.log(len(men_data)/all)
        self.female_prob = math.log(len(women_data)/all)

        self.smoothing = 1
        self.male_dict = naiveBayes(men_data, self.smoothing) #dict of probabilities based on frequency
        self.female_dict = naiveBayes(women_data, self.smoothing)

        self.intersection = self.male_dict.keys() & self.female_dict.keys()


    def is_female(self, gender_text, args, verbose=False):
        """
        Given a text, calculate the probability of it occuring in either
        category (female or not female) and return if it is more likely that this
        email is female.
        params - gender_text: data that we want to test. Each element is
                [text,tag]
                - args: args from parser
        return - boolean if female is most likely probability
        """
        #STUPID BASELINE: ALWAYS ASSUMES DOC IS FEMALE
        if self.classifier_type == 'stupid':
            return False

        tokens = word_tokenize(gender_text[0])
        t = len(tokens)

        #intl for vars for nb and bayes
        male_naive = 0 #total prob that this email is female
        female_naive = 0

        if self.classifier_type == "nb":
            for word in tokens:
                #print(word)
                #if we haven't seen this word before, make it unk
                try:
                    maleprob = self.male_dict[word] #based on frequency of word
                except:
                    maleprob = self.male_dict["<UNK>"]
                try:
                    femaleprob = self.female_dict[word]
                except:
                    femaleprob = self.female_dict["<UNK>"]

                #accumulate
                male_naive += maleprob
                female_naive += femaleprob

                # compute all probs and formulas
                total_prob_male = self.male_prob + male_naive
                total_prob_female = self.female_prob + female_naive

        elif self.classifier_type == "bayes":
            #our equation 2: we apply the Bayesian assumption
            #by multiplying weighted probabilities together
            total_prob_male = find_features(gender_text[0], 'M', args)
            total_prob_female = find_features(gender_text[0], 'F', args)

        is_female_bool = total_prob_female >= total_prob_male
        if verbose:
            print("Total prob male-authored, normalized: ", total_prob_male/t)
            print("Total prob female-authored, normalized: ", total_prob_female/t)

            if is_female_bool == True:
                print("RESULT: We predict this document is female-authored.")
            else:
                print("RESULT: We predict this document is male-authored.")

        return is_female_bool

    def tune_female(self, gender_text, mw, ww):
        """
        Tune the weights on the featuires for classifier_type bayes
        @params - gender_text: data that we want to test. Each element is
                [text,tag]
                -mw: men weights
                -ww: women weights
        @return - bool; true if female, false if male
        """
        #STUPID BASELINE: ALWAYS ASSUMES DOC IS FEMALE
        if self.classifier_type == 'stupid':
            return False

        tokens = word_tokenize(gender_text[0])
        t = len(tokens)
        #intl for vars for nb and bayes
        male_naive = 0 #total prob that this email is female
        female_naive = 0

        if self.classifier_type == "nb":
            for word in tokens:
                #if we haven't seen this word before, make it unk
                try:
                    maleprob = self.male_dict[word] #based on frequency of word
                except:
                    maleprob = self.male_dict["<UNK>"]
                try:
                    femaleprob = self.female_dict[word]
                except:
                    femaleprob = self.female_dict["<UNK>"]

                #accumulate
                male_naive += maleprob
                female_naive += femaleprob

                total_prob_male = self.male_prob + male_naive
                total_prob_female = self.female_prob + female_naive

        elif self.classifier_type == "bayes":
            #our equation 2: we apply the Bayesian assumption
            #by multiplying weighted probabilities together
            total_prob_male = tune_features(gender_text[0], 'M', mw, ww)
            total_prob_female = tune_features(gender_text[0], 'F', mw, ww)

        return total_prob_female >= total_prob_male

    def most_indicative_female(self, n):
        """
        Method to return the n most probable words for female-authored documents.
        @params - n: number of words we want to return
        @return - list of n most indicative words
        """
        probs = []
        for word in self.intersection:
            denom = math.exp(self.female_dict[word])+math.exp(self.male_dict[word])
            prob = math.log(self.female_dict[word]/math.log(denom))
            probs.append((word,prob))

        #sort list of (word, prob) tuples by probs
        probs.sort(key=itemgetter(1))
        lst = [item[0] for item in probs][:n]
        print("\nTop", n, "words with highest probability of appearing in female-authored texts:")
        print("-"*20)
        print()
        for word in lst:
            print(word)
        print()
        return lst

    def most_indicative_male(self, n):
        """
        Method to return the n most probable words for male-authored documents.
        @params - n: number of words we want to return
        @return - list of n most indicative words
        """
        probs = []
        for word in self.intersection:
            denom = math.exp(self.female_dict[word])+math.exp(self.male_dict[word])
            prob = math.log(self.male_dict[word]/math.log(denom))
            probs.append((word,prob))

        #sort list of (word, prob) tuples by probs
        probs.sort(key=itemgetter(1))
        lst = [item[0] for item in probs][:n]
        print("\nTop", n, "words with highest probability of appearing in male-authored texts:")
        print("-"*20)
        print()
        for word in lst:
            print(word)
        return lst

def test_filter(sf, test_men, test_women, args):
    """
    Given a gender filter, test how well it performs.
    @params - sf - gender filter object
    @return - n/a
    """
    #test female
    correct = 0
    for w in test_women:
        res = sf.is_female(w, args)
        if res == True: #not male
            correct += 1

    #test male
    for m in test_men:
        res = sf.is_female(m, args)
        if res == False: #is male
            correct +=1

    prop = correct/(len(test_men)+len(test_women))
    print("Our gender filter performs with %f accuracy on the test set." % prop)

def splitGender(data):
    """
    splits the data set into two lists based on gender
    @param - data: list of lists
    @returns
        - men: list of lists of only men text
        - women:  list of lists of only women text
    """
    men = []
    women = []
    for item in data:
        if item[1] == 'F':
            women.append(item)
        else:
            men.append(item)

    return men, women

def naiveBayes(gender_data, smoothing):
    """
    The orginal naive bayes, based on lab 5 log_probs()
    params  - dict: the dict of tokens
            - gender_data: running text FROM ONE SINGLE GENDER
            - smoothing: laplace smoothing variable
    return - dict of tokens to probabilities based on data

    Wrapper for calculating log probabilities of tokens. The global value of
    classifier_type determines which of the functions is used to calculate log
    probs.
    @params - gender_data: running text (either all of the male set or all of the female set)
             smoothing: laplace smoothing variable
    @return - dict of tokens to probabilities based on data
    #TODO clean this up
    """

    freq_dict = {}

    #accumulate frequencies of each word, create dict of frequencies
    #tokens = load_tokens(gender_data)
    for post in gender_data:
        tokens = []
        tokens = word_tokenize(post[0])
        for token in tokens:
            freq_dict[token.lower()] = freq_dict.setdefault(token.lower(),0) + 1
    freq_dict["<UNK>"] = 0

    v = len(freq_dict)
    w = sum(freq_dict.values())

    for word, freq in freq_dict.items():
        prob = ((freq + smoothing)/(w+smoothing*v))
        # print(prob)
        freq_dict[word] = math.log(prob)

    return freq_dict

def analysis(sf, args):
    """
    Takes hard coded data from the doc2Vec analysis of the top 10 documents
    that are most similar to M and F respectively. Then runs the classifier on
    them and prints results.
    @params
        - sf: the trained classifer object
        - args: args from the parser
    @returns None
    """
    outputM = []
    outputF = []
    # set up data for blogs or NYT
    if args.blog:
        data = read_blogs() #tenM and ten F index are based on this fcn
        tenM = [3089,1856,834,841,1693,2487,2004,2017,1580]
        tenF = [2107,1856,1693,2004,2087,2694,2567,2615,1967]
        print("\n*****Analysis of Doc2Vec Blog Top 10*****")

    if args.NYT:
        data = read_NYT() #tenM and ten F index are based on this fcn
        tenM = [22,573,392,941,60,677,107,720,604]
        tenF = [22,867,1279,713,1362,1283,1281,1259,1313]
        print("\n*****Analysis of Doc2Vec NYT Top 10*****")

    # run classifier on entries
    for idx in tenM:
        s = "The author is " + data[idx][1]
        res = sf.is_female(data[idx][0], args)
        if res == False:
            outputM.append((idx, "Male", s))
        else:
            outputM.append((idx, "Female", s))

    for idx in tenF:
        s = "The author is " + data[idx][1]
        res = sf.is_female(data[idx][0], args)
        if res == True:
            outputF.append((idx, "Female", s))
        else:
            outputF.append((idx, "Male", s))

    # generate results
    print(f"This is the top {len(tenM)} male output")
    anlyPrint(outputM)
    print(f"This is the top {len(tenF)} female output")
    anlyPrint(outputF)
    print("="*40)

    return None

def anlyPrint(L):
    """
    Helper function for analysis(sf, args). Pretty prints results
    @params - L the list of results. Each element in L should be a thruple
                (idx, gender, s)
    @returns - None
    """
    for idx, gender, s in L:
        print(f" - Entry at index {idx}: The classifier guessed {gender}."\
        f" {s}")

    return None

def test_input(filepath):
    file = open(filepath, 'r', encoding = "utf-8")
    text = file.read().strip()
    file.close()
    return [text, 'M']

def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='Runs a classifier \
                                {"stupid", naive bayes, custom} on gender data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-blog", action="count", default=0,\
                        help="This flag runs the selected classifier on blog data")
    group.add_argument("-NYT", action="count", default=0,\
                        help="This flag runs the selected classifier on NYT data")
    parser.add_argument("-type", action="store", type=str, default="nb",\
                        help="Choose your classifier type. Options: \
                            stupid, nb, and bayes.")

    parser.add_argument("--input", action="store_true",\
                        help="Use this flag if you want program to predict author gender of a document you provide")
    parser.add_argument("--top", action="store", nargs="?", type=int,\
                        default=5,\
                        help="Set the length of top different items")
    args = parser.parse_args()

    classifier_type = args.type

    #pull in data
    if args.blog:
        train, test, dev = balanced_split("blogs")

    if args.NYT:
        train, test, dev = balanced_split("NYT")

    men_data, women_data = train[0], train[1]
    men_test, women_test = test[0], test[1]
    men_dev, women_dev = dev[0], dev[1]

    #train classifer
    g = GenderFilter(men_data, women_data, 1e-5, classifier_type)
    #test the classifier
    test_filter(g, men_test, women_test, args)

    # un comments the following code to run the analysis of the data from D2V
    # analysis(g, args)

    input_filepath = "/Users/Alden/Desktop/Kiassat_Navid.txt"
    if args.input:
        #input_filepath = input("Enter ROOT filepath of document you want to test: ")
        print("\nTesting document found at ", input_filepath)
        print("-"*20)
        textlst = test_input(input_filepath)
        is_fem = g.is_female(textlst, args, verbose=True)
    print("\n")

    if args.top:
        mwords = g.most_indicative_male(args.top)
        fwords = g.most_indicative_female(args.top)
        # print(f"Top {args.top} words with highest probability of appearing in male-authored texts: {g.most_indicative_male(args.top)}\n")
        # print(f"Top {args.top} words with highest probability of appearing in female-authored texts: {g.most_indicative_female(args.top)}")

    print('Duration {}'.format(datetime.now()-start_time))

if __name__ == '__main__':
    main()
