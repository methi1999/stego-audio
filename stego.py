from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import time
import torch
import torch.nn as nn
import torch.optim
import tqdm
import torchaudio
import speech
from speech.utils.io import wave_from_array
from speech.utils.io import array_from_wave
import speech.models as models

import tensorboard_logger as tb


def inverse_delta(config, spectrogram, preproc, name):
    fs = config['fs']
    tx = torchaudio.transforms.GriffinLim(n_fft=config['win_size'] * fs // 1000,
                                          win_length=config['win_size'] * fs // 1000,
                                          hop_length=config['step_size'] * fs // 1000,
                                          window_fn=torch.hann_window)
    # undo normalisation and take inverse log i.e. exp
    spectrogram = torch.exp(preproc.invert_norm(spectrogram))
    # transform and write
    wave_from_array(tx(spectrogram), fs, name+'.wav')


def stego_spectro(config, audio_pth='tests/test1.wav', target_text=('aa', 'jh')):
    config = config['audio']
    # load model
    model, _, preproc = speech.load("ctc_best", tag="best")
    model = model.cuda() if use_cuda else model.cpu()
    # load audio file as a spectogram
    orig, _ = preproc.preprocess(pth=audio_pth)
    # pass through model to get original text
    out = model.infer_recording(orig.unsqueeze(0))[0]
    print("Decoded text in audio:", preproc.decode(out))
    delta = nn.Parameter(torch.zeros_like(orig), requires_grad=True)
    target = torch.tensor(preproc.encode(target_text)).unsqueeze(0)
    print("Encoded target:", target[0])

    # Optimizer
    optimizer = torch.optim.Adam([delta], lr=0.01)

    for e in range(1, 10000):
        to_feed = orig + delta
        inp = to_feed.unsqueeze(0)
        batch = (inp, target)
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        optimizer.step()
        if e % 50 == 0:
            d_max = delta.max().item()
            delta = torch.clamp(delta, max=d_max * 0.8).detach()
            cur_out = list(model.infer_batch(batch)[0][0])
            print("Iteration: {}, Loss: {}, cur_out: {}, d_max: {}".format(e, loss, cur_out, d_max))
            # transpose to convert it from time x freq to freq x time
            to_feed = to_feed.detach().squeeze().T
            if cur_out == target[0].tolist():
                print("Got target output")
                inverse_delta(config, to_feed, preproc, name='spectro_best')
            if e % 100 == 0:
                print("Writing audio")
                inverse_delta(config, to_feed, preproc, name='spectro_hypo')


def stego_audio(config, audio_pth='tests/timit.wav', target_text=("ix", "v", "ih", "vcl", "jh", "uw", "el", "zh", "vcl", "jh", "eh", "n")):
    # config['audio'] contains hop, window and fs params
    config = config['audio']
    # load model
    model, _, preproc = speech.load("ctc_best", tag="best")
    model = model.cuda() if use_cuda else model.cpu()
    # load audio file as a spectrogram
    orig, fs = array_from_wave(audio_pth)
    assert config['fs'] == fs
    orig_spec, _ = preproc.preprocess(audio=orig)
    # pass through model to get original text
    out = model.infer_recording(orig_spec.unsqueeze(0))[0]
    print("Decoded text in audio:", preproc.decode(out))
    # define delta
    delta = nn.Parameter(torch.zeros_like(orig), requires_grad=True).float()
    target = torch.tensor(preproc.encode(target_text)).unsqueeze(0)
    print("Encoded target:", target[0])

    thresh = orig.max()/4
    thresh_decay = 0.8
    # optimizer
    optimizer = torch.optim.Adam([delta], lr=0.01)

    for e in range(10000):
        # optimizer
        to_feed, _ = preproc.preprocess(audio=orig+delta)
        inp = to_feed.unsqueeze(0)
        batch = (inp, target)
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        optimizer.step()
        # print(delta.grad)
        # dont know why this worked
        with torch.no_grad():
            delta += delta.clamp_(min=-thresh, max=thresh) - delta

        if e % 50 == 0:
            d_max = delta.max().item()
            cur_out = list(model.infer_batch(batch)[0][0])
            print("Delta: {}".format(torch.abs(orig - delta).sum()))
            print("Iteration: {}, Loss: {}, cur_out: {}, d_max: {}, thresh: {}".format(e, loss, cur_out, d_max, thresh))
            to_write = orig + delta.clone().detach()
            wave_from_array(to_write, fs, 'aud_hypo.wav')
            if cur_out == target[0].tolist():
                print("Got target output")
                wave_from_array(to_write, fs, 'aud_best.wav')
                thresh = thresh_decay*min(thresh, delta.max())


if __name__ == "__main__":
    cfg_path = 'config.json'

    with open(cfg_path, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tb.configure(config["save_path"])
    use_cuda = torch.cuda.is_available()

    # stego_spectro(config)
    stego_audio(config)
