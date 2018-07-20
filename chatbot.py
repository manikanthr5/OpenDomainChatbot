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

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-t', '--threshold', type=int, default=20)
    parser.add_argument('-l', '--max_length', type=int, default=25)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d', '--dropout_keep_prob', type=float, default=0.2)
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

def remove_less_frequent_words(word2count, threshold, verbose=False):
    words2int = {}
    word_number = 0
    for word, count in word2count.items():
        if count >= threshold:
            words2int[word] = word_number
            word_number += 1
    if(verbose == True):
        print('Total tokens after removing less frequent words: ', len(words2int))
    return words2int

def add_tokens_to_words(words2int, verbose=False):
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
        words2int[token] = len(words2int)
    if(verbose == True):
        print('Total Tokens', len(words2int))
    return words2int

def get_inverse_dictionary(words2int, verbose=False):
    ints2word = {w_i:w for w, w_i in words2int.items()}
    if(verbose == True):
        print('Words inverse dictionary has been created!')
    return ints2word

def add_eos_to_sentences(questions, answers, verbose=False):
    for i in range(len(questions)):
        questions[i] += ' <EOS>'
    for i in range(len(answers)):
        answers[i] += ' <EOS>'
    if(verbose == True):
        print('<EOS> has been added to questions and answers')
    return (questions, answers)

def words_to_tokens(questions, answers, words2int, verbose=False):
    questions_to_int = []
    for question in questions:
        ints = []
        for word in question.split():
            if word not in words2int:
                ints.append(words2int['<OUT>'])
            else:
                ints.append(words2int[word])
        questions_to_int.append(ints)
    answers_to_int = []
    for answer in answers:
        ints = []
        for word in answer.split():
            if word not in words2int:
                ints.append(words2int['<OUT>'])
            else:
                ints.append(words2int[word])
        answers_to_int.append(ints)
    return (questions_to_int, answers_to_int)

def sort_questions_and_answers(questions, answers, max_length, verbose=False):
    sorted_questions = []
    sorted_answers = []
    for length in range(1, max_length + 1):
        for i in enumerate(questions):
            if(len(i[1]) == length):
                sorted_questions.append(questions[i[0]])
                sorted_answers.append(answers[i[0]])
    if(verbose == True):
        print('Questions and answers have been sorted according to the length of questions.')
        print('The last 5 questions are: ')
        pprint(sorted_questions[-5:])
    return (sorted_questions, sorted_answers)

def get_processed_questions_and_answers(threshold, max_length, verbose=False):
    lines, conversations = create_data_from_files(verbose=verbose)
    id2line = create_lines_dictionary(lines, verbose=verbose)
    conversations_ids = create_conversations_ids(conversations, verbose=verbose)
    questions, answers = create_questions_and_answers(id2line, conversations_ids, verbose=verbose)
    word2count = get_words_and_occurences(questions, answers, verbose=verbose)
    words2int = remove_less_frequent_words( word2count, threshold, verbose=verbose)
    words2int = add_tokens_to_words(words2int, verbose=verbose)
    ints2word = get_inverse_dictionary(words2int, verbose=verbose)
    questions, answers = add_eos_to_sentences(questions, answers, verbose=verbose)
    questions, answers = words_to_tokens(questions, answers, words2int, verbose=verbose)
    questions, answers = sort_questions_and_answers(questions, answers, max_length, verbose=verbose)
    return (questions, answers, words2int, ints2word)

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    if(verbose == True):
        print('')
    return (inputs, targets, lr, keep_prob)

def preprocess_targets(targets, words2int, batch_size):
    left_side = tf.fill([batch_size, 1], words2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1, name='preprocessed_targets')
    return preprocessed_targets

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = 'attn_dec_train')
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# To edit
# def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
#     attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
#     attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
#     training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
#                                                                               attention_keys,
#                                                                               attention_values,
#                                                                               attention_score_function,
#                                                                               attention_construct_function,
#                                                                               name = 'attn_dec_train')
#     decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
#                                                                   training_decoder_function,
#                                                                   decoder_embedded_input,
#                                                                   sequence_length,
#                                                                   scope = decoding_scope)
#     decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
#     return output_function(decoder_output_dropout)

def main(argv=sys.argv):
    '''
    The main implementation of the movie conversations chatbot
    '''
    arguments = parse_arguments(argv[1:])
    questions, answers, words2int, ints2word = get_processed_questions_and_answers(arguments['threshold'], arguments['max_length'], verbose=arguments['verbose'])


if __name__ == '__main__':
    main()
