import tensorflow as tf
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
import soundfile as psf
from utils import spectrogram
import os

RESAMPLES = 20
SAMPLE_RATE = 44100
TRAIN_PATH = "/mnt/e/Workspace/AISMentoring/F23/Seppit/MUSDB/train"
TEST_PATH = "/mnt/e/Workspace/AISMentoring/F23/Seppit/MUSDB/train"
STEMS = ["mixture", "vocals", "bass", "drums", "other"]

def multi_wave_to_spec(waves): #Look, I know this looks bad, but for some godforsaken reason a generator / list comprehension fails and I am out of ideas.
    o = (spectrogram.waveform_to_spectrogram(waves[0]),spectrogram.waveform_to_spectrogram(waves[1]),spectrogram.waveform_to_spectrogram(waves[2]),spectrogram.waveform_to_spectrogram(waves[3]),spectrogram.waveform_to_spectrogram(waves[4]))
    return tf.stack(o)

def stem_paths(folders, chunks): #Selecting blocks needs to be performed early, as shuffling full wavs or spectrograms is performance-infeasible
    allWavs = [[] for i in range(len(folders) * chunks)]
    for j,f in enumerate(folders):
        for i in range(chunks):
            offset = None
            for s in range(len(STEMS)):
                fname = f + "/" + STEMS[s] +".wav"
                if (offset==None):
                    dur = lb.get_duration(path=(fname), sr = 44100)
                    chunk_size = int((int(dur-30.0))/chunks)
                    offset = tf.random.uniform(shape = [], minval=15.0+(chunk_size*i),maxval=(15.0+(chunk_size*(i+1.0))))
                allWavs[(j*chunks) + i].append((tf.convert_to_tensor(fname, dtype = tf.string), tf.strings.as_string(offset)))
    return allWavs

def load_multi_audio(mix): #I feel dirty
    return tf.stack([tf.reshape(load_audio(mix[0][0], lb.time_to_samples(tf.strings.to_number(mix[0][1]))), [-1,1]),tf.reshape(load_audio(mix[1][0], lb.time_to_samples(tf.strings.to_number(mix[1][1]))), [-1,1]), tf.reshape(load_audio(mix[2][0], lb.time_to_samples(tf.strings.to_number(mix[2][1]))), [-1,1]), tf.reshape(load_audio(mix[3][0], lb.time_to_samples(tf.strings.to_number(mix[3][1]))), [-1,1]), tf.reshape(load_audio(mix[4][0], lb.time_to_samples(tf.strings.to_number(mix[4][1]))), [-1,1])])
    
def load_audio(filepath, start = 40, duration = 538288):
    audio, _ = psf.read(filepath.numpy(), start = start, frames = duration, dtype = "float32")
    audio_tensor = tf.convert_to_tensor(audio)
    padding = [[0, tf.maximum(524288 - tf.shape(audio_tensor)[0],0)]]
    audio_tensor = tf.pad(tf.reduce_sum(audio_tensor,1), padding, 'CONSTANT', constant_values=0)
    audio_tensor = audio_tensor[:538288]
    return audio_tensor

def build_evaluation_dataset():
    folders = [x[0] for x in os.walk(TEST_PATH)][1:]
    paths_set = stem_paths(folders, 1)
    dataset = tf.data.Dataset.from_tensor_slices(paths_set)
    dataset = dataset.map(lambda x: tf.py_function(load_multi_audio, [x], Tout=[tf.float32]))
    dataset = dataset.map(multi_wave_to_spec)
    dataset = dataset.map(spectrogram.trim)
    dataset = dataset.batch(4)
    dataset = dataset.repeat()
    return dataset

def build_train_dataset():
    folders = [x[0] for x in os.walk(TRAIN_PATH)][1:]
    paths_set = stem_paths(folders, RESAMPLES)
    dataset = tf.data.Dataset.from_tensor_slices(paths_set)
    dataset = dataset.shuffle(
        100000, seed=110, reshuffle_each_iteration=True
    )
    dataset = dataset.map(lambda x: tf.py_function(load_multi_audio, [x], Tout=[tf.float32]))
    dataset = dataset.map(multi_wave_to_spec)
    dataset = dataset.map(spectrogram.trim)
    dataset = dataset.batch(4)
    dataset = dataset.repeat()
    return dataset

    """
    coollist = [[tf.reshape(load_audio(x), [-1, 1]) for x in y] for y in paths_set[0]]
    
    #coollist = [tf.reshape(load_audio(x), [-1, 1]) for x in paths_set[0]]
    dataset = tf.data.Dataset.from_tensor_slices(coollist)
    #dataset = dataset.map(lambda file_path: (load_audio(file_path)))
    for e in dataset.take(1):
        print(e)
    #dataset = dataset.map(lambda waveform: (spectrogram.waveform_to_spectrogram(waveform)))
    for e in dataset.take(5):
        tmp = spectrogram.gain_to_db(e.numpy())
        tmp = np.squeeze(tmp, axis=-1)
        plt.pcolormesh(tmp)
        plt.title("test")
        plt.savefig("test.png")
        
    #   tmp = spectrogram.gain_to_db(tmp)
    #   tmp = spectrogram.change_speed(tmp,1.5)
    #   tmp = np.squeeze(tmp, axis=-1)
    #   plt.pcolormesh(tmp)
    #   plt.title("test")
    #   plt.savefig("test2.png")"""
    


if __name__ == "__main__":
    build_dataset()