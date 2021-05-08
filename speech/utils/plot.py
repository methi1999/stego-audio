import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import pickle

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