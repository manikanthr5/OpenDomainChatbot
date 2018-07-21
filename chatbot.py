"""
Chatbot project main file.
Building a chatbot with Deep NLP.
"""

# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
import tqdm
import sys
import argparse
from pprint import pprint
from seq2seq import *
from data_processing import *

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-t', '--threshold', type=int, default=20)
    parser.add_argument('-m', '--max_length', type=int, default=25)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d', '--dropout_keep_prob', type=float, default=0.2)
    parser.add_argument('-l', '--lines_file', required=False, default='./data/movie_lines.txt')
    parser.add_argument('-c', '--conversations_file', required=False, default='./data/movie_conversations.txt')
    try:
        arguments = parser.parse_args(args=args)
    except:
        parser.print_help()
        sys.exit(0)
    arguments = vars(arguments)
    return arguments

def main(argv=sys.argv):
    '''
    The main implementation of the movie conversations chatbot
    '''
    arguments = parse_arguments(argv[1:])
    questions, answers, words2int, ints2word = get_processed_questions_and_answers(arguments['lines_file'], arguments['conversations_file'], arguments['threshold'], arguments['max_length'], verbose=arguments['verbose'])

if __name__ == '__main__':
    main()
