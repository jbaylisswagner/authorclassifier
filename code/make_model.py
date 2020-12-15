"""
Build and train a doc2vec model of given corpus using "M" and "F" tags to see
if our model can determine any meaningful distinctions between the two classes.
Authors: Adriana Knight and Naomi Park
11/13
"""
from sys import argv
from infuse_data import read_blogs, read_NYT
import nltk
import numpy as np
nltk.download('punkt')
from random import choice
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def process(data, random):
    """
    Process data into a doc2vec model that creates a vector representation for
    each individual written entry against a one item list of tags corresponding
    to sex marker.
    Heavily pulled from
    [https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5]
    params - data: list of lists of (content, gender) where we want to tokenize
                   the content and create a single vector representing the data.
             random: flag to determine if we will build model using real tags
                     or randomly assigned ones
    return - model: doc2vec representation of all data. Will spit out an
                    appropriate numerical vector when prompted using field
                    docvecs.
    """
    # format data into tagged document, build and train model
    if not random:
        tagged_data = [TaggedDocument(words=data[i][0],
                                      tags=["%d"%i, data[i][1]]) for i in range(len(data))]
    else:
        tagged_data = [TaggedDocument(words=data[i][0],
                                      tags=["%d"%i, choice(["M", "F"])]) for i in range(len(data))]
    #print(tagged_data)
    vec_size = 20
    max_epochs = 50
    alpha = 0.025

    model = Doc2Vec(size=vec_size,alpha=alpha,min_alpha=0.00025,min_count=1,dm=1)
    model.build_vocab(tagged_data)
    for epoch in tqdm(range(max_epochs)):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Trained d2v model.")

    # find top 10 most indicative texts for each category
    most_m = model.docvecs.most_similar("M")
    most_f = model.docvecs.most_similar("F")

    print("Top 10 Most Male Docs:")
    for i in range(len(most_m)):
        print("%8s: %f" % (most_m[i][0], most_m[i][1]))
    print("Top 10 Most Female Docs:")
    for j in range(len(most_f)):
        print("%8s: %f" % (most_f[j][0], most_f[j][1]))

    #calculate similarity between vectors representing male and female tags
    m = model.docvecs["M"]
    f = model.docvecs["F"]
    abstract_m = np.reshape(m, (1, -1))
    abstract_f = np.reshape(f, (1, -1))

    similarity = cosine_similarity(abstract_m, abstract_f)
    print("The similarity between our gender vectors is", similarity)

    return model

def main():
    #command line parse
    if len(argv) != 3:
        print("Expected usage: python3 make_model [blog/nyt] [t/r]")
        exit()

    if argv[1] == "blog":
        data = read_blogs(verbose=False)
    else:
        data = read_NYT(verbose=False)

    if argv[2] == "t":
        r_flag = False
    else:
        r_flag = True

    tokenized_data = [[word_tokenize(item[0]), item[1]] for item in data]
    print("Starting doc2vec modeling...")
    doc_model = process(tokenized_data, r_flag)

if __name__ == '__main__':
    main()
