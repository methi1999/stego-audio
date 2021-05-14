import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import pickle


def pesq_loss_edit_iter(pickle_path, title):
    loss, pesq, edit = pickle.load(open(pickle_path, 'rb'))
    # pesq
    iters = [x[0] for x in pesq]
    wb, nb = [x[1] for x in pesq], [x[2] for x in pesq]
    plt.plot(iters, wb, color='b', label='wide-band', marker='x')
    plt.plot(iters, nb, color='r', label='narrow-band', marker='x')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("PESQ score")
    plt.grid(True)
    plt.legend()
    plt.show()
    # edit
    # plt.plot([x[0] for x in edit], [x[1] for x in edit], color='g')
    # plt.title(title)
    # plt.xlabel("Iterations")
    # plt.ylabel("Edit distance")
    # plt.grid(True)
    # plt.legend()
    # plt.show()


def pesq_loss_edit_iter_noise(pickle_path1, pickle_path2, title):
    loss, pesq, edit = pickle.load(open(pickle_path1, 'rb'))
    # pesq
    iters = [x[0] for x in pesq]
    wb, nb = [x[1] for x in pesq], [x[2] for x in pesq]
    plt.plot(iters, wb, color='b', label='Proper text (Sr. No. 1)', marker='x')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("PESQ score")
    plt.grid(True)

    # # edit
    # iters, edit = [x[0] for x in pesq], [x[1] for x in pesq]
    # plt.plot(iters, edit, color='b', label='Without Noise', marker='x')
    # plt.title(title)
    # plt.xlabel("Iterations")
    # plt.ylabel("Edit")
    # plt.grid(True)

    loss, pesq, edit = pickle.load(open(pickle_path2, 'rb'))
    # pesq
    iters = [x[0] for x in pesq]
    wb, nb = [x[1] for x in pesq], [x[2] for x in pesq]
    plt.plot(iters, wb, color='r', label='Random text (Sr. No. 2)', marker='x')
    plt.legend()
    plt.show()


def plot_noise():
    _, orig = read('rec_3.wav')
    _, new = read('rec_3_asr.wav')
    print(new.max())
    new = new*(2**15)
    delta = new - orig
    print(orig, new)
    plt.plot(orig, alpha=0.5, label='Original')
    plt.plot(delta, label='Perturbation')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Original and Perturbation\nRecording 3")

    plt.show()


if __name__ == '__main__':
    # pesq_loss_edit_iter('../../examples/TIMIT_test/test_nuclear.pkl', '')
    pesq_loss_edit_iter_noise('../../examples/TIMIT_test/test_nuclear.pkl', '../../examples/TIMIT_test/test_rand.pkl',
                              'TIMIT - random vs proper text')
