"""
11/11/20
This file provides the functions that read in the data sets
Jake Chanenson
"""
import csv
import os
import re
import sys
from copy import deepcopy
from random import shuffle

def read_blogs(verbose=False):
    """
    Reads in blog posts from csv file and returns 1548 men and 1548 women blog posts
    @param - (optional) verbose: set to true if you want summary
    @ret - list of lists. each sub list has data at element 0 and gender at
          element 1
    """
    rows = []
    mcnt, wcnt, tcnt, mc, wc = [0,0,0,0,0] #counters

    # generate path
    currPath = os.path.dirname(__file__)
    fullPath = os.path.join(currPath, '..', 'data', 'blog-gender-dataset.csv')


    reader = csv.reader(open(fullPath, encoding = 'utf-8-sig'))

    for row in reader:
        if row[1].strip() in ['M', 'm']:
            mcnt += 1
            if mcnt < 1549 and row[0] != '':
                rows.append([row[0].strip(), 'M'])
                mc+=1
        elif row[1].strip() in ['F', 'f']:
            wcnt += 1
            if wcnt < 1549 and row[0] != '':
                rows.append([row[0].strip(), 'F'])
                wc+=1
        else:
            #intentional. don't have an else condition any more
            # need the if and elif to ensure we don't get an empty entry
            pass

    tcnt = mcnt+wcnt
    if verbose:
        print("Sucesfully read in %d M and %d W blog entries" %(mc, wc))
        print("\n*****Blog Summary*****\nTotal Blog Entries: %d\nTotal M:"\
        f" %d\nTotal F: %d" %(tcnt,mcnt,wcnt))
        print("\n*****Imported Blog Dataset Summary*****\nTotal Entries:"\
        f" %d\nTotal M: %d\nTotal F: %d" %((wc+mc),mc,wc))

    return rows

def split_blogs(n=.90, static=True):
    """
    Splits up the blog data set and returns it as a training and testing data
    @param
        - n: on the domian [0,1] specifies what percentage should be training
        - static: triggers shuffing of data set before split
    @returns
        - training data
        - testing data
    """
    #input validation
    if n<=0 or n>1:
        print("Please input a value between 0 and 1")
        sys.exit()

    data = read_blogs(False)
    data_copy = deepcopy(data)

    if static != True:
        shuffle(data_copy)
    train_amt = int(len(data_copy)*n)

    return data_copy[:train_amt], data_copy[train_amt:]

def read_NYT(verbose=False):
    """
    Reads in NYT opnion colmns gendred men or women
    @param - (optional) verbose: set to true if you want summary
    @ret- list of lists. each sub list has data at element 0 and gender at
          element 1
    """
    mcnt, wcnt, ucnt, mc, wc = [0,0,0,0,0] #counters
    retLst = []
    unkownLst = []

    # generate relative path
    currPath = os.path.dirname(__file__)
    fullPath = os.path.join(currPath, '..', 'data/nyt')

    #get file names
    files = os.listdir(fullPath)
    nytPaths = []
    for file in files:
        nytPaths.append(os.path.join(fullPath, file))

    for nytPath in nytPaths:
        f = open(nytPath, 'r', encoding = "utf-8")

        #handle unknown authors gender
        if re.match(".*/data/nyt/\d*_u\w*", nytPath):
            unkownLst.append([f.read().strip(), 'U'])
            ucnt += 1
            continue
        #check to see if the author is M or F
        if re.match(".*/data/nyt/\d*_m\w*", nytPath):
            gender = 'M'
            mcnt += 1
            if mcnt < 837:
                retLst.append([f.read().strip(), 'M'])
                mc += 1
        elif re.match(".*/data/nyt/\d*_f\w*", nytPath):
            gender = 'F'
            wcnt += 1
            if wcnt < 837:
                retLst.append([f.read().strip(), 'F'])
                wc += 1

    if verbose:
        print("Sucesfully read in %d M and %d W NYT articles" %(mc, wc))
        print("\n*****NYT Summary*****\nTotal Blog Entries: %d\nTotal M:"\
        f" %d\nTotal F: %d\nTotal Other %d" %(len(nytPaths),mcnt,wcnt,ucnt))
        print("\n*****Imported NYT Dataset Summary*****\nTotal Entries:"\
        f" %d\nTotal M: %d\nTotal F: %d" %((wc+mc),mc,wc))

    return retLst

def split_NYT(n=.90, static=True):
    """
    Splits up the NYT data set and returns it as a training and testing data
    @param
        - n: on the domian [0,1] specifies what percentage should be training
        - static: triggers shuffing of data set before split
    @returns
        - training data
        - testing data
    """
    #input validation
    if n<=0 or n>1:
        print("Please input a value between 0 and 1")
        sys.exit()

    data = read_NYT(False)
    data_copy = deepcopy(data)

    if static != True:
        shuffle(data_copy)

    train_amt = int(len(data_copy)*n)
    test_amt = int(len(data_copy)*(1-n))

    return data_copy[test_amt:], data_copy[:test_amt]

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

def balanced_split(type, verbose = False):
    """
    Produces train, dev, and test sets (.80,.10,.10 of the data respectively)
    each data set has 50/50 gender parity
     @params
        - type: either "NYT" or "blogs"
    @return: a list of three lists: train, test, dev.
        train[0] - men
        train[1] - women
        dev[0] - men
        dev[1] - women
        test[0] - men
        test[1] - women
    """
    if type == "NYT":
        data = read_NYT()
    elif type == "blogs":
        data = read_blogs()
    else:
        print("Please choose 'NYT' or 'blogs'")
        return None, None, None

    men, women = splitGender(data)
    total_amt = len(men)
    train_amt = int(total_amt*.80)
    dev_amt = int(total_amt*.10)

    train = [[],[]]
    for i in range(train_amt):
        train[0].append(men.pop())
    for j in range(train_amt):
        train[1].append(women.pop())

    dev = [[],[]]
    for k in range(dev_amt):
        dev[0].append(men.pop())
    for w in range(dev_amt):
        dev[1].append(women.pop())

    test = [[],[]]
    test[0] = men
    test[1] = women

    if verbose:
        print("\n*****Men %s Data*****"%(type))
        print(f"Train is {len(train[0])} ({len(train[0])/total_amt:.2f})%\n"\
        f"Dev is {len(dev[0])} ({len(dev[0])/total_amt:.4f})%\nTest is {len(test[0])}"\
        f" ({len(test[0])/total_amt:.4f})%\n")
        print(f"Total: {total_amt}; Train: {len(train[0])} "\
        f"+ Dev: {len(dev[0])} + Test: {len(test[0])}"\
        f" = {len(train[0])+len(dev[0])+len(test[0])} ")

        print("\n*****Women %s Data*****"%(type))
        print(f"Train is {len(train[1])} ({len(train[1])/total_amt:.2f})%\n"\
        f"Dev is {len(dev[1])} ({len(dev[1])/total_amt:.4f})%\nTest is {len(test[1])}"\
        f" ({len(test[0])/total_amt:.4f})%\n")
        print(f"Total: {total_amt}; Train: {len(train[1])} "\
        f"+ Dev: {len(dev[1])} + Test: {len(test[1])}"\
        f" = {len(train[1])+len(dev[1])+len(test[1])} ")

    return train, dev, test

def load_posneg(verbose = False):
    """
    Loads in the csv sentiments.csv and creates two dicts
    @param - verbose: set to true if you want summary
    @returns
        - posDict: dict of all pos words
        - negDict: dict of all neg words
    """
    posDict = {}
    negDict = {}
    #dict[token] = dict.setdefault(token,0) + 1
    # generate path
    currPath = os.path.dirname(__file__)
    fullPath = os.path.join(currPath, '..', 'data', 'sentiments.csv')

    reader = csv.reader(open(fullPath, encoding = 'utf-8-sig'))

    for row in reader:
        if row[1] == 'positive':
            posDict[row[0]] = posDict.setdefault(row[0], 0) + 1
        else: #must be neg
            negDict[row[0]] = negDict.setdefault(row[0],0) + 1

    if verbose:
        totalWords = len(posDict)+len(negDict)
        s = f"\n*****Sentiments Summary*****\nTotal number words:" \
        f"{totalWords}\nNumber of pos words: {len(posDict)}" \
        f" ({(len(posDict)/totalWords)*100:.2f}%)\nNumber of neg words:" \
        f" {len(negDict)} ({(len(negDict)/totalWords)*100:.2f}%)"
        print(s)


    return posDict, negDict

def main():
    print("This file contains functions that import the data.")
    #Un-comment below to run various import functions for data summaries
    # foo = read_blogs(True)
    # bar = read_NYT(True)
    # pos, neg = load_posneg(True)
    # balanced_split("NYT", True)
    # balanced_split("blogs", True)

if __name__ == '__main__':
    main()
