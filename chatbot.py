"""
Chatbot project main file.
Building a chatbot with Deep NLP.
"""

# Importing the libraries
import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
from seq2seq import *
from data_processing import *
from nlp_utils import *
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-V', '--verbose', action='store_true')
    # Hyper Parameters
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-th', '--threshold', type=int, default=20)
    parser.add_argument('-sl', '--sequence_length', type=int, default=25)
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-rs', '--rnn_size', type=int, default=1024)
    parser.add_argument('-nl', '--num_layers', type=int, default=3)
    parser.add_argument('-ee', '--encoding_embedding_size', type=int, default=1024)
    parser.add_argument('-de', '--decoding_embedding_size', type=int, default=1024)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=0.9)
    parser.add_argument('-mlr', '--minimum_learning_rate', type=float, default=0.0001)
    parser.add_argument('-kp', '--keep_probability', type=float, default=0.5)
    parser.add_argument('-vr', '--validation_set_ratio', type=float, default=0.15)
    # File locations
    parser.add_argument('-lf', '--lines_file', required=False, default='./data/movie_lines.txt')
    parser.add_argument('-cf', '--conversations_file', required=False, default='./data/movie_conversations.txt')
    try:
        arguments = parser.parse_args(args=args)
    except:
        parser.print_help()
        sys.exit(0)
    arguments = vars(arguments)
    return arguments

def model_inputs(verbose = False):
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    if(verbose == True):
        print('model inputs placeholders have been created!')
    return (inputs, targets, lr, keep_prob)

def build_model(batch_size = None,
                threshold = None,
                sequence_length = None,
                epochs = None,
                rnn_size = None,
                num_layers = None,
                encoding_embedding_size = None,
                decoding_embedding_size = None,
                learning_rate = None,
                learning_rate_decay = None,
                minimum_learning_rate = None,
                keep_probability = None,
                validation_set_ratio = None,
                lines_file = None,
                conversations_file = None,
                verbose = None):
    tf.reset_default_graph()
    session = tf.InteractiveSession()
    inputs, targets, lr, keep_prob = model_inputs(verbose)
    questions, answers, questionswords2int, answerswords2int, answersints2word = \
                                                    get_processed_questions_and_answers(lines_file,
                                                    conversations_file,
                                                    threshold,
                                                    sequence_length,
                                                    verbose)
    sequence_length = tf.placeholder_with_default(sequence_length, None, name = 'sequence_length')
    input_shape = tf.shape(inputs)
    print('Shape of inputs, targets after preprocess_targets: ', np.shape(inputs), np.shape(targets))
    print('Shape of inputs, targets after preprocess_targets: ', np.shape(tf.reverse(inputs, [-1])), np.shape(targets))
    training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                           targets,
                                                           keep_prob,
                                                           batch_size,
                                                           sequence_length,
                                                           len(answerswords2int),
                                                           len(questionswords2int),
                                                           encoding_embedding_size,
                                                           decoding_embedding_size,
                                                           rnn_size,
                                                           num_layers,
                                                           questionswords2int)
    print('Shape of training_questions, test_predictions: ', np.shape(training_predictions), np.shape(test_predictions))
    with tf.name_scope('optimization'):
        print('Shape of training predictions, targets: ', np.shape(training_predictions), np.shape(targets))
        loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                      targets,
                                                      tf.ones([input_shape[0], sequence_length]))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss_error)
        print('Shape of gradients: ', np.shape(gradients))
        clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
        optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

    # Training and Validation Split
    training_questions, training_answers, validation_questions, validation_answers = \
                                get_training_validation_data(questions,
                                                             answers,
                                                             validation_set_ratio)
    print('Shape of training: questions-> {}, answers->{}'.format(np.shape(training_questions), np.shape(training_answers)))
    print('Shape of validation: questions-> {}, answers->{}'.format(np.shape(validation_questions), np.shape(validation_answers)))

    # Training
    batch_index_check_training_loss = 100
    batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
    total_training_loss_error = 0
    list_validation_loss_error = []
    early_stopping_check = 0
    early_stopping_stop = 1000
    checkpoint = "chatbot_weight.ckpt"
    session.run(tf.global_variables_initializer())
    print('Varibles have been initialized!')
    for epoch in tqdm(range(1, epochs + 1)):
        for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in \
                                enumerate(split_into_batches(training_questions,
                                                             training_answers,
                                                             questionswords2int,
                                                             answerswords2int,
                                                             batch_size)):
            print('Shape of padded_question_in_batch, padded_answers_in_batch: ', np.shape(padded_questions_in_batch), np.shape(padded_answers_in_batch))
            starting_time = time.time()
            print('inputs: ', type(padded_questions_in_batch))
            print('targets: ', type(padded_answers_in_batch))
            print('learning_rate: ', type(learning_rate))
            print('sequence_length: ', type(padded_questions_in_batch.shape[1]))
            _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error],
                                            feed_dict = {
                                            inputs: padded_questions_in_batch,
                                            targets: padded_answers_in_batch,
                                            lr: learning_rate,
                                            sequence_length: padded_questions_in_batch.shape[1],
                                            keep_prob: keep_probability,
                                            })
            total_training_loss_error += batch_training_loss_error
            ending_time  = time.time()
            batch_time = ending_time - starting_time
            if(batch_index % batch_index_check_training_loss == 0):
                print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on {} Batches: {:d} seconds'.format(
                                            epoch,
                                            epochs,
                                            batch_index,
                                            len(training_questions) // batch_size,
                                            batch_index_check_training_loss,
                                            total_training_loss_error // batch_index_check_training_loss,
                                            int(batch_index * batch_index_check_training_loss)
                                            ))
                total_training_loss_error = 0
            if(batch_index % batch_index_check_validation_loss == 0 and batch_index > 0):
                total_validation_loss_error = 0
                starting_validation_time = time.time()
                for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in \
                                                enumerate(split_into_batches(validation_questions,
                                                                             validation_answers,
                                                                             questionswords2int,
                                                                             answerswords2int,
                                                                             batch_size)):
                    batch_validation_loss_error = session.run(loss_error,
                                                    {
                                                    inputs: padded_questions_in_batch,
                    TeslaV100                                targets: padded_answers_in_batch,
                                                    lr: learning_rate,
                                                    sequence_length: padded_questions_in_batch.shape[1],
                                                    keep_prob: 1
                                                    })
                    total_validation_loss_error += batch_validation_loss_error
                    ending_validation_time  = time.time()
                    batch_validation_time = endingvalidation__time - startingvalidation__time
                    average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
                    print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_validation_time), ))
                    learning_rate *= learning_rate_decay
                    if learning_rate < minimum_learning_rate:
                        learning_rate = minimum_learning_rate
                    list_validation_loss_error.append(average_validation_loss_error)
                    if average_validation_loss_error <= min(list_validation_loss_error):
                        print('I speak better now')
                        early_stopping_check = 0
                        saver = tf.train.Saver()
                        saver.save(session, checkpoint)
                    else:
                        print('Sorry, I do not speak better, I need to practice more!')
                        early_stopping_check += 1
                        if early_stopping_check >= early_stopping_stop:
                            break
        if early_stopping_check >= early_stopping_stop:
            print("My apologies, I cannot speak better anymore, This is the best I can do!")
            break
    print('Game over!')

def convert_string2int(question, words2int):
    question = clean_text(question)
    return [words2int.get(word, words2int['<OUT>']) for word in question.split()]

def chat_with_bot():
    checkpoint = './checkpoint.ckpt'
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, checkpoint)
    while True:
        question = input("You: ")
        if(question == 'Goodbye'):
            break
        else:
            question = convert_string2int(question, questionswords2int)
            question = question + [questionswords2int['<PAD>']] * (20 - len(question))
            fake_batch = np.zeros((batch_size, 20))
            fake_batch[0] = question
            predicted_answer = session.run(test_predictions, {input: fake_batch, keep_prob: 0.5})[0]
            answer = ''
            for i in np.argmax(predicted_answer, 1):
                if answersints2word[i] == 'i':
                    token = 'I'
                elif answersints2word[i] == '<EOS>':
                    token = '.'
                elif answersints2word[i] == '<OUT>':
                    token = 'out'
                else:
                    token = ' ' + answersints2word[i]
                answer += token
                if token == '.':
                    break
            print('Chatbot:', answer)


def main(argv=sys.argv):
    '''
    The main implementation of the movie conversations chatbot
    '''
    arguments = parse_arguments(argv[1:])
    build_model(**arguments)
    chat_with_bot()

if __name__ == '__main__':
    main()
