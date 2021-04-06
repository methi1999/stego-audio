from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import json
import os
import random
import tqdm

from speech.utils import data_helpers
from speech.utils.io import wav_duration

WAV_EXT = "wv"  # using wv since NIST took wav
# TEST_SPEAKERS = [ # Core test set from timit/readme.doc
#     'mdab0', 'mwbt0', 'felc0', 'mtas1', 'mwew0', 'fpas0',
#     'mjmp0', 'mlnt0', 'fpkt0', 'mlll0', 'mtls0', 'fjlm0',
#     'mbpm0', 'mklt0', 'fnlp0', 'mcmj0', 'mjdh0', 'fmgd0',
#     'mgrt0', 'mnjm0', 'fdhc0', 'mjln0', 'mpam0', 'fmld0']

# TEST_SPEAKERS = [
#      'DAB0', 'WBT0', 'ELC0', 'TAS1', 'WEW0', 'PAS0',
#      'JMP0', 'LNT0', 'PKT0', 'LLL0', 'TLS0', 'JLM0',
#      'BPM0', 'KLT0', 'NLP0', 'CMJ0', 'JDH0', 'MGD0',
#      'GRT0', 'NJM0', 'DHC0', 'JLN0', 'PAM0', 'MLD0']

TEST_SPEAKERS = [
    'FAKS0', 'MDAB0', 'MJSW0', 'FCMR0', 'MABW0', 'MBJK0',
    'FCMH0', 'MBDG0', 'MBWM0', 'FADG0', 'MBNS0', 'MDLS0',
    'FASW0', 'MAHH0', 'MBPM0', 'FDRW0', 'MCMJ0', 'MDSC0',
    'FCAU0', 'MCHH0', 'MDLF0', 'FCMH1', 'MAJC0', 'MDAW1']


def load_phone_map():
    with open("phones.60-48-39.map", 'r') as fid:
        lines = (l.strip().split() for l in fid)
        lines = [l for l in lines if len(l) == 3]
    m60_48 = {l[0]: l[1] for l in lines}
    m48_39 = {l[1]: l[2] for l in lines}
    return m60_48, m48_39


def load_transcripts(path):
    pattern = os.path.join(path, "*/*/*.PHN")
    m60_48, _ = load_phone_map()
    files = glob.glob(pattern)
    print("Load Transcript:", path, files)
    # Standard practice is to remove all "sa" sentences
    # for each speaker since they are the same for all.
    filt_sa = lambda x: os.path.basename(x)[:2] != "sa"
    files = filter(filt_sa, files)
    data = {}
    for f in tqdm.tqdm(files, desc="Loading Transcripts"):
        with open(f) as fid:
            lines = (l.strip() for l in fid)
            phonemes = (l.split()[-1] for l in lines)
            phonemes = [m60_48[p] for p in phonemes if p in m60_48]
            data[f] = phonemes
    return data


def split_by_speaker(data, dev_speakers=50):
    def speaker_id(f):
        return os.path.basename(os.path.dirname(f))

    speaker_dict = collections.defaultdict(list)
    for k, v in data.items():
        speaker_dict[speaker_id(k)].append((k, v))
    speakers = list(speaker_dict.keys())
    print("speakers:", speakers)
    not_present = 0
    for t in TEST_SPEAKERS:
        if t in speakers:
            speakers.remove(t)
        else:
            not_present += 1
    print(f"{not_present * 100 / len(TEST_SPEAKERS)}% not present")
    random.shuffle(speakers)
    dev = speakers[:dev_speakers]
    dev = dict(v for s in dev for v in speaker_dict[s])
    test = dict(v for s in TEST_SPEAKERS for v in speaker_dict[s])
    return dev, test


def convert_to_wav(path):
    data_helpers.convert_full_set(path, "*/*/*/*.*.wav",
                                  new_ext=WAV_EXT,
                                  use_avconv=False)


def build_json(data, path, set_name):
    basename = set_name + os.path.extsep + "json"
    with open(os.path.join(path, basename), 'w+') as fid:
        for k, t in tqdm.tqdm(data.items()):
            # print("Here", k, t)
            # wave_file = os.path.splitext(k)[0] + ".WAV" + os.path.extsep + WAV_EXT
            wave_file = os.path.splitext(k)[0] + ".wav"
            # wave_file = k + os.path.extsep + WAV_EXT
            dur = wav_duration(wave_file)
            datum = {'text': t,
                     'duration': dur,
                     'audio': wave_file}
            json.dump(datum, fid)
            fid.write("\n")


if __name__ == "__main__":
    # ../dataset
    # parser = argparse.ArgumentParser(description="Preprocess Timit dataset.")
    # parser.add_argument("output_directory", help="Path where the dataset is saved.")
    # args = parser.parse_args()
    # path = os.path.join(args.output_directory, "TIMIT")

    root = 'dataset'
    path = os.path.join(root, "TIMIT")

    path = os.path.abspath(path)
    if not os.path.exists(path):
        print("Making directory")
        os.makedirs(path)

    # print("Converting files from NIST to standard wave format...")
    # convert_to_wav(path)

    print("Preprocessing train")
    train = load_transcripts(os.path.join(path, "TRAIN"))
    # print(train)
    build_json(train, path, "TRAIN")

    print("Preprocessing dev")
    transcripts = load_transcripts(os.path.join(path, "TEST"))
    dev, test = split_by_speaker(transcripts)
    build_json(dev, path, "DEV")

    print("Preprocessing test")
    build_json(test, path, "TEST")
