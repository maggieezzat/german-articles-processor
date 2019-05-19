# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import listdir, remove
from os.path import isfile, join

import codecs
import os
import sys
import unicodedata

import re

from absl import app as absl_app
import pandas
from six.moves import urllib
import tensorflow as tf

import string
import collections


test="/home/maggie/10kGNAD/test.csv"
train="/home/maggie/10kGNAD/train.csv"
test_res="/home/maggie/10kGNAD/test_res.tsv"


def test_fn(test_dir=test, train_dir=train):

    num_train=0
    num_test=0

    train_list = []
    test_list = []

    label_dict={
        "Web": 0,
        "Panorama": 1,
        "International": 2,
        "Wirtschaft": 3,
        "Sport": 4,
        "Inland": 5,
        "Etat": 6,
        "Wissenschaft": 7,
        "Kultur": 8,
    }

    with open(test_dir, 'r') as f:
        while True:
            line = f.readline()
            line_copy = line
            if not line:
                break
            line = line.split(';')
            label = line[0]
            label_ID = label_dict.get(label)
            article = line_copy[len(label)+1:-1]
            num_test+=1
            test_list.append( (num_test, label, label_ID, article) )


    df = pandas.DataFrame(
            data=test_list, columns=["Example_ID", "Label", "Label_ID", "Article"]
        )
    output_file = "/home/maggie/articles_test.tsv"
    df.to_csv(output_file, index=False, sep="\t")

    with open(train_dir, 'r') as f:
        while True:
            line = f.readline()
            line_copy = line
            if not line:
                break
            line = line.split(';')
            label = line[0]
            label_ID = label_dict.get(label)
            article = line_copy[len(label)+1:-1]
            num_train+=1
            train_list.append( (num_train, label, label_ID, article) )

    
    df = pandas.DataFrame(
            data=train_list, columns=["Example_ID", "Label", "Label_ID", "Article"]
        )
    output_file = "/home/maggie/articles_train.tsv"
    df.to_csv(output_file, index=False, sep="\t")


def check_test_results(test_dir=test, test_result=test_res):

    num_test_examples=0
    num_correct_pred=0

    label_dict={
        "Web": 0,
        "Panorama": 1,
        "International": 2,
        "Wirtschaft": 3,
        "Sport": 4,
        "Inland": 5,
        "Etat": 6,
        "Wissenschaft": 7,
        "Kultur": 8,
    }

    with open(test_dir, 'r') as f:
        with open(test_result, 'r') as res_f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.split(';')
                correct_label = line[0]
                correct_label_ID = label_dict.get(correct_label)
                #print(correct_label)
                #print(correct_label_ID)
                num_test_examples+=1

                predicted_value = res_f.readline()
                predicted_value = predicted_value.split()[-1]

                #print(predicted_value)
                #break

                if int(correct_label_ID) - int(predicted_value) == 0:
                    num_correct_pred+=1

    print("Number of Correct Predictions: "+str(num_correct_pred)+"/"+str(num_test_examples))
    with open("/home/maggie/test_result_percentage.txt", 'w') as f:
        f.write("Number of Correct Predictions: "+str(num_correct_pred)+"/"+str(num_test_examples) + "\n")
        f.write("Percentage: " + str((num_correct_pred/num_test_examples) * 100))


        



def main(_):
    test_fn()
    #check_test_results()
   

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    absl_app.run(main)