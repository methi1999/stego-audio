import os
import pickle
import torch
import numpy as np
from scipy.io.wavfile import read, write

MODEL = "model"
PREPROC = "preproc"


"""
preproc is the preprocessor which contains char_to_id and its inverse mapping, normalisation statistics, etc.
It is dumped so that during inference, the above variables can be retrieved
"""


def get_names(path, tag):
    tag = tag + "_" if tag else ""
    model = os.path.join(path, tag + MODEL)
    preproc = os.path.join(path, tag + PREPROC)
    return model, preproc


def save(model, opti, preproc, path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    torch.save((model, opti), model_n)
    with open(preproc_n, 'wb') as fid:
        pickle.dump(preproc, fid)


def load(path, tag=""):
    model_n, preproc_n = get_names(path, tag)
    model, opti = torch.load(model_n, map_location=torch.device('cpu'))
    with open(preproc_n, 'rb') as fid:
        preproc = pickle.load(fid)
    return model, opti, preproc


def array_from_wave(file_name):
    samp_rate, audio = read(file_name)
    audio = torch.tensor(audio).float()
    # Todo: Fix this. Current implementation is jugaad
    if np.abs(audio).max() > 10:  # integers
        audio /= 32768
    return audio, samp_rate


def wave_from_array(sig, fs, pth):
    write(pth, fs, np.array(sig))


def wav_duration(file_name):
    samp_rate, audio = read(file_name)
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration

