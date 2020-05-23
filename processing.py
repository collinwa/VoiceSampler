import random
import os
import pydub 
import numpy as np

def read(f, normalized=True):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


def concat_raw_signal(singer, instrument): 
    min_value = singer.shape[0]

    if (singer.shape[0] > instrument.shape[0]):
        min_value = instrument.shape[0]

    min_value = min_value - (min_value % 32)
    cut1 = singer[:min_value,:]
    cut2 = instrument[:min_value,:]
    
    # deal with when the samples are too short
    if cut1.shape[0] < min_value or cut2.shape[0] < min_value:
        return None, None
    # just average out the normalized values
    new_signal = (4.0*cut1/5.0) + cut2/5.0
    # convert back into 16-bit signal
    return np.array(new_signal, dtype=np.float64), np.array(cut1, dtype=np.float64)
