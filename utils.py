import torch
import torch.nn.functional as F
import random
import os
import pydub 
import numpy as np


def crossent(y_hat, y):
	return F.binary_crossentropy(y_hat, y)


def read(f, duration=0.5, rate=44100, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))    

    y = y.T

    cut_length = int(duration * rate)

    if normalized:
        return a.frame_rate, torch.tensor(np.float32(y[np.newaxis,:,:cut_length]) / 2**15)
    else:
        return a.frame_rate, torch.tensor(np.float32(y[np.newaxis,:,:cut_length]))
