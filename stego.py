import json
import random
import pickle
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torchaudio
import speech
from speech.utils.io import wave_from_array
from speech.utils.io import array_from_wave
import speech.models as models
from speech.utils.score import pesq_score, compute_cer
import argparse
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


def stego_audio(config_full, audio_pth, add_noise, inp_target, is_text, dump_suffix=''):
    # config['audio'] contains hop, window and fs params
    if add_noise:
        dump_suffix += '_noise'
    config = config_full['audio']
    hypo_path = audio_pth[:-4] + '_hypo.wav'
    pkl_path = audio_pth[:-4] + '_{}.pkl'.format(dump_suffix)
    # load model
    model, _, preproc = speech.load("ctc_best", tag="best")
    # Freeze model params. Even if we don't, it doesnt matter since optimiser has only been passed delta as param
    for parameter in model.parameters():
        parameter.requires_grad = False
    # transfer to cuda
    model = model.cuda() if use_cuda else model.cpu()
    # load audio file as a spectrogram
    orig, fs = array_from_wave(audio_pth)
    assert config['fs'] == fs
    orig_spec, _ = preproc.preprocess(audio=orig)
    # pass through model to get original text
    out = model.infer_recording(orig_spec.unsqueeze(0))[0]
    print("Decoded text in audio: {}\nDecoded Sequence: {}".format(preproc.decode(out), out))
    # define delta
    delta = nn.Parameter(torch.zeros_like(orig), requires_grad=True).float()
    if is_text:
        target = torch.tensor(preproc.encode(inp_target)).unsqueeze(0)
        target_list = target[0].tolist()
    else:
        target = torch.tensor(inp_target).unsqueeze(0)
        target_list = list(inp_target)
    print("Target to encode:", target[0])

    edit_dists = [(0, compute_cer([(target_list, out)]))]
    # parameters
    thresh = orig.max()/3
    thresh_decay = 0.75
    check_every = 50
    num_iter = 10000
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
            print("Delta: {}".format(torch.abs(orig - delta).sum()))
            d_max = delta.max().item()
            print("Iteration: {}, Loss: {}, cur_out: {}, d_max: {}, thresh: {}".format(e, loss, cur_out, d_max, thresh))
            to_write = orig + delta.clone().detach()
            wave_from_array(to_write, fs, hypo_path)
            # store edit distance
            edit_dists.append((e, compute_cer([(target_list, cur_out)])))

            if cur_out == target_list:
                print("Got target output")
                best_path = audio_pth[:-4] + '_' + '_'.join([str(e), dump_suffix]) + '.wav'
                wave_from_array(to_write, fs, best_path)
                thresh = thresh_decay*min(thresh, d_max)
                pesq_nb = pesq_score(orig.numpy(), to_write.numpy(), fs, 'nb')
                pesq_wb = pesq_score(orig.numpy(), to_write.numpy(), fs, 'wb')
                print('PESQ score: {} {}'.format(pesq_nb, pesq_wb))
                logs.append((e, pesq_nb, pesq_wb))

                # dump data
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

        if e % 200 == 0:
            with open(pkl_path, 'wb') as f:
                pickle.dump((losses, logs, edit_dists), f)


    # dump data
    with open(pkl_path, 'wb') as f:
        pickle.dump((losses, logs, edit_dists), f)

    return losses, logs, edit_dists


def get_random(length):
    return np.random.randint(low=0, high=48, size=length).tolist()


if __name__ == "__main__":
    cfg_path = 'config.json'

    with open(cfg_path, 'r') as fid:
        config = json.load(fid)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a")
    args = parser.parse_args()

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tb.configure(config["save_path"])
    # use_cuda = torch.cuda.is_available()
    use_cuda = False

    # nuclear strike authorised
    s1 = "n uw k l iy er sil s t r ay k sil ao th er ay z d sil".split(' ')
    # shyam, how is your semester exchange
    s2 = 'sh y aa m sil hh aw ih z sil y ao r sil s ah m eh s t er sil ih k s ch ey n jh sil'.split(' ')
    # Hey Alexa, add a TV to my shopping list
    s3 = 'hh ey ah l eh k s ah sil ae d ah t iy v iy t uw sil m ay sh aa p ih ng l ih s t'.split(' ')
    # code red
    s4 = 'k ow d sil r eh d'.split(' ')

    # make sets of examples
    egs = [('recordings/test.wav', s1, 'nuclear', True), ('recordings/test.wav', get_random(15), 'rand', False),
           ('recordings/walter.wav', s2, 'shyam', True), ('recordings/walter.wav', get_random(15), 'rand', False),
           ('recordings/destroyer.wav', s4, 'code_red', True), ('recordings/destroyer.wav', get_random(40), 'rand', False)]

    if args.a == 'all':
        for rec_pth, to_enc, suffix, is_text in egs:
            # stego_spectro(config)
            stego_audio(config, audio_pth=rec_pth, add_noise=False, inp_target=to_enc, is_text=is_text, dump_suffix=suffix)
    else:
        idx = int(args.a)
        stego_audio(config, audio_pth=egs[idx][0], add_noise=False, inp_target=egs[idx][1], is_text=egs[idx][3], dump_suffix=egs[idx][2])


    # automatic speech recognition
    # target = (
    #     'sil', 'ao', 't', 'ah', 'm', 'ae', 't', 'ih', 'k', 'sil', 's', 'p', 'iy', 'ch', 'sil', 'r', 'eh', 'k', 'ah',
    #     'g', 'n', 'uh', 'sh', 'ah', 'n', 'sil'
    # )
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
