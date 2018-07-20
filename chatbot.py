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

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    try:
        arguments = parser.parse_args(args=args)
    except:
        parser.print_help()
        sys.exit(0)
    arguments = vars(arguments)
    return arguments

def create_data_from_files(verbose=False):
    ### Importing Dataset
    with open('./data/movie_lines.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().split('\n')
    with open('./data/movie_conversations.txt', 'r', encoding='utf-8', errors='ignore') as f:
        conversations = f.read().split('\n')
    if(verbose == True):
        print('{} lines and {} conversations have been extracted.'.format(len(lines), len(conversations)))
    return (lines, conversations)

def create_lines_dictionary(lines, verbose=False):
    '''
    Creating a dictionary that maps each line and its id
    '''
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

def create_conversations_ids(conversations, verbose=False):
    '''
    To get a list of conversation ids for each conversation by removing brackets, quotes and spaces.
    '''
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(','))
    if(verbose == True):
        print('{} conversations ids have been created.'.format(len(conversations_ids)))
    return conversations_ids

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " wourld", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-{}#/@;:\[\]<>{}+=|.,\\]", "", text)
    return text

def create_questions_and_answers(id2line, conversations_ids, verbose=False):
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range(len(conversation) - 1):
            questions.append(clean_text(id2line[conversation[i]]))
            answers.append(clean_text(id2line[conversation[i+1]]))
    if(verbose == True):
        print('{} quesions and {} answers have been created.'.format(len(questions), len(answers)))
    return (questions, answers)

def get_words_and_occurences(questions, answers, verbose=False):
    word2count = {}
    for sentence in questions + answers:
        for word in sentence.split():
            if word in word2count:
                word2count[word] += 1
            else:
                word2count[word] = 1
    if(verbose == True):
        print('{} words have been found.'.format(len(word2count)))
    return word2count

def main(argv=sys.argv):
    '''
    The main implementation of the movie conversations chatbot
    '''
    arguments = parse_arguments(argv[1:])
    lines, conversations = create_data_from_files(verbose=arguments['verbose'])
    id2line = create_lines_dictionary(lines, verbose=arguments['verbose'])
    conversations_ids = create_conversations_ids(conversations, verbose=arguments['verbose'])
    questions, answers = create_questions_and_answers(id2line, conversations_ids, verbose=arguments['verbose'])
    word2count = get_words_and_occurences(questions, answers, verbose=arguments['verbose'])

if __name__ == '__main__':
    main()
