from VoiceMixer import MixVoiceData
from processing import *
import numpy as np

def averageSixteen(track):
    SIXTEENSECONDS = 16 * 44100
    length = track.shape[0] - (track.shape[0] % SIXTEENSECONDS)
    trim = track[:length,:]
    stack = trim.reshape(length // SIXTEENSECONDS, SIXTEENSECONDS * 2)
    averages = np.mean(stack, axis=0)
    return np.ravel(averages.reshape(SIXTEENSECONDS, 2))

def theta(track1, track2):
    return np.arccos(np.dot(track1, track2) / (np.linalag.norm(track1) * np.linalg.norm(track2)))

