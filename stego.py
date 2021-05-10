import json
import time
import random
import pickle
import torch
import torch.nn as nn
import torch.optim
import argparse

import torchaudio
import speech
from speech.utils.io import wave_from_array
from speech.utils.io import array_from_wave
import speech.models as models
from speech.utils.score import pesq_score, compute_cer

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
            delta = torch.clamp(delta, max=d_max * 0.9).detach()
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


def stego_audio(config_full, add_noise, audio_pth='tests/timit.wav',
                target_text=('hh', 'ah', 'l', 'ow', 'sil', 'w', 'er', 'l', 'd')):
    # config['audio'] contains hop, window and fs params
    config = config_full['audio']
    hypo_path = audio_pth[:-4] + '_hypo.wav'
    # load model
    model, _, preproc = speech.load("ctc_best", tag="best")
    # Freeze model params. Even if we don't, it doesnt matter since optimiser has only been passed delta as param
    for parameter in model.parameters():
        parameter.requires_grad = False
    # transfer to cuda
    model = model.cuda() if use_cuda else model.cpu()
    # load audio file as a spectrogram
    orig, fs = array_from_wave(audio_pth)
    print(fs)
    assert config['fs'] == fs
    orig_spec, _ = preproc.preprocess(audio=orig)
    # pass through model to get original text
    out = model.infer_recording(orig_spec.unsqueeze(0))[0]
    print("Decoded text in audio: {}\nDecoded Sequence: {}".format(preproc.decode(out), out))
    # define delta
    delta = nn.Parameter(torch.zeros_like(orig), requires_grad=True).float()
    target = torch.tensor(preproc.encode(target_text)).unsqueeze(0)
    target_list = target[0].tolist()
    print("Target to encode:", target[0])
    edit_dists = [compute_cer([(target_list, out)])]
    # parameters
    thresh = orig.max()/3
    thresh_decay = 0.75
    check_every = 50
    num_iter = 30000
    # RMS/3 ToDo: convert to SNR based calculation
    if add_noise:
        noise_sigma = torch.sqrt(torch.mean(orig**2)).item()/3
        print("Noise sigma:", noise_sigma)
    # optimizer
    optimizer = torch.optim.Adam([delta], lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995, verbose=True)
    lr_step = 50
    logs, losses = [], []

    for e in range(1, num_iter):
        # optimizer
        audio_to_feed = orig+delta
        if add_noise:
            audio_to_feed += torch.normal(mean=0, std=noise_sigma, size=audio_to_feed.size())
        to_feed, _ = preproc.preprocess(audio=orig+delta)
        inp = to_feed.unsqueeze(0)
        batch = (inp, target)
        optimizer.zero_grad()
        loss = model.loss(batch)

        if e % check_every == 0:
            cur_out = list(model.infer_batch(batch)[0][0])
            global start_time
            print(start_time)
            print("Time taken for", check_every, "iterations:", time.time() - start_time)
            print("Delta: {}".format(torch.abs(orig - delta).sum()))
            d_max = delta.max().item()
            print("Iteration: {}, Loss: {}, cur_out: {}, d_max: {}, thresh: {}".format(e, loss, cur_out, d_max, thresh))
            to_write = orig + delta.clone().detach()
            wave_from_array(to_write, fs, hypo_path)
            # store edit distance
            edit_dists.append((e, compute_cer([(target_list, cur_out)])))
            print(edit_dists)
            if cur_out == target_list:
                print("Got target output")
                best_path = audio_pth[:-4] + '_enc_' + str(e) + '.wav'
                wave_from_array(to_write, fs, best_path)
                thresh = thresh_decay*min(thresh, d_max)
                pesq_nb = pesq_score(orig.numpy(), to_write.numpy(), fs, 'nb')
                pesq_wb = pesq_score(orig.numpy(), to_write.numpy(), fs, 'wb')
                print('PESQ score: {} {}'.format(pesq_nb, pesq_wb))
                logs.append((e, pesq_nb, pesq_wb))

                # dump data
                pkl_path = audio_pth[:-4] + '_data.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump((losses, logs, edit_dists), f)

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        # dont know why this worked
        with torch.no_grad():
            delta += delta.clamp_(min=-thresh, max=thresh) - delta

        if e % lr_step == 0:
            lr_scheduler.step()

    return losses, logs, edit_dists


# def pesq_vs_iter():

def load_targets(path):
    with open(path, "r") as f:
        lines = f.readlines()
    phones = [i.strip() for i in lines]
    phones = (" ".join(phones)).strip().split(" ")
    return phones

# start_time = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", help="Path to audio file on which to encrypt text")
    parser.add_argument("--target", help='Path to target phone sequence file')
    parser.add_argument("--noise", help='Add gaussian noise: 0/1 value', default=0)
    args = parser.parse_args()

    cfg_path = 'config.json'

    with open(cfg_path, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tb.configure(config["save_path"])
    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    global start_time
    start_time = time.time()
    # target = ('sil', 'ao', 't', 'ah', 'm', 'ae', 't', 'ih', 'k', 'sil', 's', 'p', 'iy', 'ch', 'sil', 'r', 'eh', 'k', 'ah',
              # 'g', 'n', 'uh', 'sh', 'ah', 'n', 'sil')
    target = load_targets(args.target)
    print(target)
    # stego_spectro(config)
    stego_audio(config, add_noise=int(args.noise), audio_pth=args.audio_path, target_text=target)
    print("Time taken by script:", time.time() - start_time)
    # can get phase from audio by setting power=None in spectrogram but GriffinLim does not accept it as input
    # so no point, otherwise we find other functions/libraries which can invert spectrogram

    # config = config['audio']
    # window_size, step_size = config['win_size'], config['step_size']
    # sample_rate = config['fs']
    # nperseg = int(window_size * sample_rate / 1e3)
    # noverlap = int(step_size * sample_rate / 1e3)
    # spec_mag = torchaudio.transforms.Spectrogram(n_fft=nperseg,
    #                                          win_length=nperseg,
    #                                          hop_length=noverlap,
    #                                          window_fn=torch.hann_window)
    # spec_comp = torchaudio.transforms.Spectrogram(n_fft=nperseg,
    #                                              win_length=nperseg,
    #                                              hop_length=noverlap,
    #                                              window_fn=torch.hann_window,
    #                                              power=None)
    # tx = torchaudio.transforms.GriffinLim(n_fft=config['win_size'] * sample_rate // 1000,
    #                                       win_length=config['win_size'] * sample_rate // 1000,
    #                                       hop_length=config['step_size'] * sample_rate // 1000,
    #                                       window_fn=torch.hann_window,
    #                                       power=None)
    # orig, fs = array_from_wave('tests/timit.wav')
    # s = spec_comp(orig)
    # print(s.size())
    # out = tx(s)
    # print(out)
    # wave_from_array(out, fs, 'inverse.wav')
    # # print(torch.sum(s**2, dim=2))
    # # print(spec_mag(orig))
    # # torch.log(spec(audio) + eps)
