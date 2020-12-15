#!/bin/bash

python3 tune_classifier.py -NYT -epoch 500 1>> bash_run_long_NYT.txt

python3 tune_classifier.py -NYT -epoch 1000 1>> bash_run_long_NYT.txt

python3 tune_classifier.py -NYT -epoch 10000 1>> bash_run_long_NYT.txt




