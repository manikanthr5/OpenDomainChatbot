"""
Main Data Pre-Processing File
"""

# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time
import tqdm
import sys
import argparse

def create_data_from_files(lines_file, conversations_file, verbose=False):
    ### Importing Dataset
    with open(lines_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().split('\n')
    with open(conversations_file, 'r', encoding='utf-8', errors='ignore') as f:
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
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
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

def remove_less_frequent_words(word2count, threshold, verbose=False):
    questionswords2int = {}
    answerswords2int = {}
    word_number = 0
    for word, count in word2count.items():
        if count >= threshold:
            questionswords2int[word] = word_number
            answerswords2int[word] = word_number
            word_number += 1
    if(verbose == True):
        print('Total tokens after removing less frequent words: ', len(answerswords2int))
    return (questionswords2int, answerswords2int)

def add_tokens_to_words(questionswords2int, answerswords2int, verbose=False):
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
        questionswords2int[token] = len(questionswords2int) + 1
        answerswords2int[token] = len(answerswords2int) + 1
    if(verbose == True):
        print('Total Tokens', len(answerswords2int))
    return (questionswords2int, answerswords2int)

def get_inverse_dictionary(answerswords2int, verbose=False):
    answersints2word = {w_i:w for w, w_i in answerswords2int.items()}
    if(verbose == True):
        print('Words inverse dictionary of answers has been created!')
    return answersints2word

def add_eos_to_sentences(answers, verbose=False):
    for i in range(len(answers)):
        answers[i] += ' <EOS>'
    if(verbose == True):
        print('<EOS> has been added to answers')
    return answers

def words_to_tokens(questions, answers, questionswords2int, answerswords2int, verbose=False):
    questions_to_int = []
    for question in questions:
        ints = []
        for word in question.split():
            if word not in questionswords2int:
                ints.append(questionswords2int['<OUT>'])
            else:
                ints.append(questionswords2int[word])
        questions_to_int.append(ints)
    answers_to_int = []
    for answer in answers:
        ints = []
        for word in answer.split():
            if word not in answerswords2int:
                ints.append(answerswords2int['<OUT>'])
            else:
                ints.append(answerswords2int[word])
        answers_to_int.append(ints)
    return (questions_to_int, answers_to_int)

def sort_questions_and_answers(questions, answers, sequence_length, verbose=False):
    sorted_questions = []
    sorted_answers = []
    for length in range(1, sequence_length + 1):
        for i in enumerate(questions):
            if(len(i[1]) == length):
                sorted_questions.append(questions[i[0]])
                sorted_answers.append(answers[i[0]])
    if(verbose == True):
        print('Questions and answers have been sorted according to the length of questions.')
    return (sorted_questions, sorted_answers)

def get_processed_questions_and_answers(lines_file, conversations_file, threshold, sequence_length, verbose=False):
    lines, conversations = create_data_from_files(lines_file, conversations_file, verbose=verbose)
    id2line = create_lines_dictionary(lines, verbose=verbose)
    conversations_ids = create_conversations_ids(conversations, verbose=verbose)
    questions, answers = create_questions_and_answers(id2line, conversations_ids, verbose=verbose)
    word2count = get_words_and_occurences(questions, answers, verbose=verbose)
    questionswords2int, answerswords2int = remove_less_frequent_words(word2count, threshold, verbose=verbose)
    questionswords2int, answerswords2int = add_tokens_to_words(questionswords2int, answerswords2int, verbose=verbose)
    answersints2word = get_inverse_dictionary(answerswords2int, verbose=verbose)
    answers = add_eos_to_sentences(answers, verbose=verbose)
    questions, answers = words_to_tokens(questions, answers, questionswords2int, answerswords2int, verbose=verbose)
    questions, answers = sort_questions_and_answers(questions, answers, sequence_length, verbose=verbose)
    return (questions, answers, questionswords2int, answerswords2int, answersints2word)
