# Hear Me If You Can!

This repository contains the code for our project on audio steganography. Most of our work is based on
[this](https://isca-speech.org/archive/Interspeech_2020/abstracts/1294.html) paper.
This work was done as part of [CS753: Automatic Speech Recognition](https://www.cse.iitb.ac.in/~pjyothi/cs753/index.html) at IIT Bombay.

A major chunk of the repo has been forked from [here](https://github.com/awni/speech).

Contributors: Samyak Shah, Rishabh Dahale and Mithilesh Vaidya

## Installation

We recommend creating a virtual environment and installing the python
requirements there.

```
virtualenv <path_to_your_env>
source <path_to_your_env>/bin/activate
pip install -r requirements.txt
```

## Directory structure

*ctc_best*: Trained ASR models

*examples*: a few examples which are mentioned in the presentation. Each recording has a folder which contains:
1. name.wav: original clean recording e.g. walter.wav
2. name_encoded_text.wav: perturbed recording for the encoded text
3. name_encoded_text.pkl: pickle file containing loss and PESQ score as a function of the number of iterations

*speech*: main codebase which contains the ASR model and preprocessing steps

*final_presentation.pptx*: a brief presentation of our project

*stego.py*: contains the actual stego algorithm

*train.py*: file for training the ASR model

## Running the code

*stego.py* contains the both algorithms for calculating the perturbation in time-domain and spectral-domain

It takes as input a path to the audio recording and a list of phones to encode.
