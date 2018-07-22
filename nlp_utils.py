'''
Main utility functions
'''
import numpy as np
import tensorflow as tf


def preprocess_targets(targets, words2int, batch_size):
    left_side = tf.fill([batch_size, 1], words2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

def apply_padding(batch_of_sequences, words2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [words2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

def split_into_batches(questions, answers, questionswords2int, answerswords2int, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch

def get_training_validation_data(questions, answers, validation_set_ratio):
    training_validation_split = int(len(questions) * validation_set_ratio)
    training_questions = questions[training_validation_split:]
    training_answers = answers[training_validation_split:]
    validation_questions = questions[:training_validation_split]
    validation_answers = answers[:training_validation_split]
    return (training_questions, training_answers, validation_questions, validation_answers)
