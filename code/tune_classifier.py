"""
A short script to try and computationally come up with the best tuning paramters
12/9/2020
J. Chanenson
"""

import sys
import argparse
import random

# things we wrote
from genderc import *
from features import *
from infuse_data import balanced_split

def tune_filter(sf, test_men, test_women):
    """
    Given a gender filter, test how well it performs.
    params - sf - gender filter object
    return - n/a
    """
    mw = {
        "pos":1,
        "neg":1,
        "emotion":1,
        "excl":1,
        "commas":1,
        "periods":1,
        "questions":1,
        "lexical":1
    }
    ww = {
        "pos":1,
        "neg":1,
        "emotion":1,
        "excl":1,
        "commas":1,
        "periods":1,
        "questions":1,
        "lexical":1
    }

    for key,val in mw.items():
        mw[key] = float(random.uniform(1e-5,3))

    for key, val in ww.items():
        ww[key] = float(random.uniform(1e-5,3))

    #test female
    correct = 0
    for w in test_women:
        res = sf.tune_female(w, mw, ww)
        if res == True: #not male
            correct += 1

    #test male
    for m in test_men:
        res= sf.tune_female(m, mw, ww)
        if res == False: #is male
            correct +=1

    prop = correct/(len(test_men)+len(test_women))
    return prop, mw, ww


def main():
    parser = argparse.ArgumentParser(description='Tunes params for genderc.py')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-blog", action="count", default=0,\
                        help="This flag tunes params for blog data")
    group.add_argument("-NYT", action="count", default=0,\
                        help="This flag tunes params for NYT data")
    parser.add_argument("-epoch", action="store", type=int, default=1,\
                        help="set the number of epochs to run")
    args = parser.parse_args()

    classifier_type = "bayes"

    #import data
    if args.blog:
        train, test, dev = balanced_split("blogs")

    if args.NYT:
        train, test, dev = balanced_split("NYT")

    men_data, women_data = train[0], train[1]
    men_test, women_test = test[0], test[1]
    men_dev, women_dev = dev[0], dev[1]

    print("Creating classifier", file=sys.stderr)
    g = GenderFilter(men_data, women_data, 1e-5, classifier_type)

    max_acc = 0.0
    max_mw = {}
    max_ww = {}
    accm = 0
    for i in range(args.epoch):
        accm+=1
        print("Running epoch %d of %d"%(accm,args.epoch), file=sys.stderr)
        accuracy, mw, ww = tune_filter(g, men_dev, women_dev)
        if accuracy > max_acc:
            max_acc = accuracy
            max_mw = mw
            max_ww = ww

    print(f"\nEpochs: {args.epoch}\n")
    print(f"Our gender filter performs with {max_acc} accuracy on the dev set.")
    print(f"Men Weights: {max_mw}\n\nWomen Weights: {max_ww}")
    print("-"*10)


if __name__ == '__main__':
    main()
