# authorclassifier

This was a group final project for our CS65 Natural Language Processing course. We built a naive Bayes baseline and Bayesian classifier in order to see if we could train the latter to identify author gender. We used two different data sets: a set of NYT Opinions articles, and a set of miscellaneous blog posts. Each had an equal number of men and women. In addition, we built a doc2vec system, which created vectors for each document in the corpora and determined which were most likely male- or female-authored.

## What is in our repository
* code
    - `infuse_data.py` - contains methods to read in and clean the datasets.
    - `human_readable.py` - simple script to grab individual texts for human inspection.
    - `make_model.py` - builds and trains a doc2vec model of the corpora.
    - `genderc.py` - gender classifier that has integrated features along with naive bayes, and a "stupid" classifier.
    - `features.py` - helper functions for `genderc.py` that extract features and calculate feature related probabilities.
    - `tune_classifier.py` - tunes the weights for the custom gender classifier found in `genderc.py`
    - `long_run_blog.sh` - bash script that runs several instances of `tune_classifier.py`on blog data with increasingly larger epochs.
        - saves output to `bash_run_long_blog.txt`
    - `long_run_NYT.sh` - bash script that runs several instances of `tune_classifier.py`on NYT data with increasingly larger epochs.
        - saves output to `bash_run_long_NYT.txt`

* data
    - citations.md - file that contains the citations for the data we found
    - sentiments.csv - a list of words tagged as either positive or negative
    - Mukherjee-and-Liu_blog-gender-dataset.xlsx - the orginal blog dataset
    - blog-gender-dataset.csv - a csv version of the Mukherjee and Liu dataset. This is the version that we use to read in data.
    - NYTIMES.tar.gz - zipped version of the nyt datset

* results
  - /classifier_results - a folder containing `.txt` files of the various classifier accuracy scores
  - /doc2vec_results - folder containing `.txt` files of doc2vec results and rankings, as well as unpacked copies of the most gendered documents as identified in trials
  - /data_import_info - a folder containing `.txt` summaries of the data imports
  - /tuning_data - a folder containing `.txt` logs of the tuning runs
  - d2V-classifier_anly.txt - a `.txt` file with the classifier outputs of the doc2vec top 10 most male and most female documents

## Set Up
* Install dependencies
  - To run our code, you must install these packages using pip. Here's how to do it in terminal:
    - `pip install {PACKAGENAME}`
  - If you do not yet have pip, here's how to install:
    - `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
    - `python get-pip.py`

* Dependencies
    - `nltk`
    - `numpy`
    - `sklearn`
    - `gensim`
    - `tqdm`

* Unpack NYT Data
    - go to `\data`
    - type `$ tar -xf NYTIMES.tar.gz`
    - type `$ mv clean nyt`
        - the gitignore file is looking to ignore a sub folder of data called nyt



## How to run
`$ python3 genderc.py [-h] (-blog | -NYT) [-type TYPE] [--top [TOP]]`

`$ make_model.py [blog/nyt] [t/r]`

