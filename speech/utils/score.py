from pesq import pesq
import editdistance


def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
            predicted sequence pairs.

    Returns the CER for the full set.
    """
    dist = sum(editdistance.eval(label, pred) for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total


def pesq_score(ref, deg, rate, band):
    """
    ref: reference/clean audio
    deg: degraded version
    rate: sampling rate
    band: wb or nb (wide-band or narrow-band)
    """
    return pesq(rate, ref, deg, band)


if __name__ == '__main__':
    from scipy.io.wavfile import read
    fs, a = read('../../recordings/destroyer.wav')
    _, b = read('../../recordings/destroyer_enc_oldd.wav')
    print(pesq_score(a, b, fs, band='nb'))
