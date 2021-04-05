from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tqdm

from speech.utils import convert


def convert_full_set(path, pattern, new_ext="wav", **kwargs):
    """
    To convert from other audio formats to wav
    """
    pattern = os.path.join(path, pattern)
    audio_files = glob.glob(pattern)
    for af in tqdm.tqdm(audio_files):
        # split across extension e.g. /ff/ff/ff/.wav will be split into /ff/ff/ff and wav
        base, ext = os.path.splitext(af)
        # os.path.extsep is the separator e.g. /
        wav = base + os.path.extsep + new_ext
        convert.to_wave(af, wav, **kwargs)




