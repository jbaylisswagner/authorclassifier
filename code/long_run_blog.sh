#!/bin/bash

python3 tune_classifier.py -blog -epoch 500 1>> bash_run_long_blog.txt

python3 tune_classifier.py -blog -epoch 1000 1>> bash_run_long_blog.txt

python3 tune_classifier.py -blog -epoch 10000 1>> bash_run_long_blog.txt




