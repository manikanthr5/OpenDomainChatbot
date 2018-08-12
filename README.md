# Open Domain Chatbot
This repository contains code for a generative type chatbot created using [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) with [attention mechanism](https://en.wikipedia.org/wiki/Recurrent_neural_network).

## Table of Contents
- [Introduction](#introduction)
  - [Retrieval-Based vs. Generative Chatbots](#retrieval-based-vs-generative-chatbots)
  - [Prerequesites](#prerequesites)
- [Installation Steps](#installation-steps)
- [Data Used](#data-used)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Chat with Bot](#chat-with-bot)
- [Current Status](#current-status)
- [Future Prospects](#future-prospects)

## Introduction
Now a days, the importance of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) and [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing) has been improving at a very remarkable rate. The main progress is seen in areas like [Automatic Speech Recognition](https://en.wikipedia.org/wiki/Speech_recognition), [Neural Machine Translation](https://en.wikipedia.org/wiki/Machine_translation), [Image Recognition](https://en.wikipedia.org/wiki/Computer_vision), [Self Driving Cars](https://en.wikipedia.org/wiki/Autonomous_car), [Chatbots](https://en.wikipedia.org/wiki/Chatbot), [Recomender Systems](https://en.wikipedia.org/wiki/Recommender_system) and [other areas](https://en.wikipedia.org/wiki/Deep_learning#Applications).

This is a Generative Type Chatbot developed using Recurrent Neural Networks with Attension Mechanism in TensorFlow Deep Learning Framework.


### Retrieval-Based vs. Generative Chatbots

Some Background On Chatbots taken from [WildML](http://www.wildml.com) article on [Retrieval-Based vs. Generative Models](http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/).

**Retrieval-based models** (easier) use a repository of predefined responses and some kind of heuristic to pick an appropriate response based on the input and context. The heuristic could be as simple as a rule-based expression match, or as complex as an ensemble of Machine Learning classifiers. These systems don’t generate any new text, they just pick a response from a fixed set.

**Generative models** (harder) don’t rely on pre-defined responses. They generate new responses from scratch. Generative models are typically based on Machine Translation techniques, but instead of translating from one language to another, we “translate” from an input to an output (response).

### Prerequesites

I am creating this project to apply the skills I learnt while working on the course [Deep Learning for Natural Language Processing](http://web.stanford.edu/class/cs224n/) taught by Stanford Professors, [Christopher Manning](https://nlp.stanford.edu/~manning/) and [Richard Socher](https://www.socher.org/). I suggest you go through this course in order to understand about Bi-directional Recurrent Neural Networks, Embeddings and Attention Mechanism in Seq2Seq models.

## Installation Steps
In order to use this repository to create a chatbot, you first need have [Python 3.5](https://docs.python.org/3.5/) or more and need to install some Python Packages. I suggest you to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for this purpose. Run the following commands in your favorite projects directory.
```
cd path/to/your/folder
git clone https://github.com/manikanthr5/MovieChatbot.git
cd MovieChatbot
virtualenv -p python3.5 env
source env/bin/activate
pip install -r requirements.txt
```

Now you are ready to work with the Chatbot.

## Data Used
I am using [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) for the purpose of training the model. Please download the data from this [link](http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip) and extract **movie_conversations.txt** and **movie_lines.txt** to ./data folder. Other files in this zip folder are optional and please read through them to understand more about the data.

## Project Structure
The project is divided into **3** parts.

## Model Training

## Chat with Bot

## Current Status

## Further Steps
