import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as siowav

def mp3_to_wav(src, dst):
  out = AudioSegment.from_mp3(src)
  out.export(dst, format='wav')

def wav_to_mp3(src, dst):
  out = AudioSegment.from_wav(src)
  out.export(dst, format='mp3')

def load_sparse_matrix(wavfile):
  rate, data = siowav.read(wavfile)
  norm_signal = np.array(data, dtype=np.float64)
  return rate, norm_signal

rate1, singer_signal = load_sparse_matrix('./wavs/test_singer.wav')
rate2, instrumental_signal = load_sparse_matrix('./wavs/test_wav.wav')

print(rate1)
print(singer_signal.shape)
print(instrumental_signal.shape)
print(rate2)

def concat_raw_signal(singer, instrument):
    min_value = singer.shape[0]
    if (singer.shape[0] > instrument.shape[0]):
        min_value = instrument.shape[0]
        cut1 = singer[:min_value,:]
        cut2 = instrument[:min_value,:]

    # just average out the normalized values
    new_signal = (4.0*cut1/5.0) + cut2/5.0
    # convert back into 16-bit signal
    return np.array(new_signal, dtype=np.int16)


if __name__ == '__main__':
    new_signal = concat_raw_signal(singer_signal, instrumental_signal)

    siowav.write('./test_sample.wav', rate1, new_signal)
    wav_to_mp3('./test_sample.wav', './test_sample.mp3')

