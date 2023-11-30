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
TEST_PATH = "/mnt/e/Workspace/AISMentoring/F23/Seppit/MUSDB/testing/"
STEMS = ["mixture", "vocals", "bass", "drums", "other"]

def multi_wave_to_spec(waves): #Look, I know this looks bad, but for some godforsaken reason a generator / list comprehension fails and I am out of ideas.
    o = (spectrogram.waveform_to_spectrogram(waves[0]),spectrogram.waveform_to_spectrogram(waves[1]),spectrogram.waveform_to_spectrogram(waves[2]),spectrogram.waveform_to_spectrogram(waves[3]),spectrogram.waveform_to_spectrogram(waves[4]))
    return tf.stack(o)

def stem_paths(folders):
    allWavs = [[] for i in range(len(folders))]
    for i,f in enumerate(folders):
        offset = None
        for s in range(len(STEMS)):
            allWavs[i].append(f + "/" + STEMS[s] +".wav")
    return allWavs

def load_multi_audio(mix, chunks): #I feel dirty
    dur = lb.get_duration(path=(mix[0].numpy()), sr = 44100)
    samples = []
    chunk_size = int((int(dur-30.0))/chunks)
    for i in range(chunks):
        start = tf.random.uniform(shape = [], minval=15.0+(chunk_size*i),maxval=(15.0+(chunk_size*(i+1.0))))
        tmp = (tf.stack([tf.reshape(load_audio(mix[0], lb.time_to_samples(start)), [-1,1]),tf.reshape(load_audio(mix[1], lb.time_to_samples(start)), [-1,1]), tf.reshape(load_audio(mix[2], lb.time_to_samples(start)), [-1,1]), tf.reshape(load_audio(mix[3],lb.time_to_samples(start)), [-1,1]), tf.reshape(load_audio(mix[4],lb.time_to_samples(start)), [-1,1])]))
        samples.append(tmp)
    return tf.stack(samples)

def load_audio(filepath, start = 40, duration = 524288):
    audio, _ = psf.read(filepath.numpy(), start = start, frames = duration, dtype = "float32")
    audio_tensor = tf.convert_to_tensor(audio)
    return audio_tensor

def build_evaluation_dataset():
    folders = [x[0] for x in os.walk(TEST_PATH)][1:]
    paths_set = stem_paths(folders)
    dataset = tf.data.Dataset.from_tensor_slices(paths_set)
    dataset = dataset.map(lambda x: tf.py_function(load_multi_audio, [x, RESAMPLES], Tout=[tf.float32]))
    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    dataset = dataset.map(multi_wave_to_spec)
    dataset = dataset.map(spectrogram.trim_freq)
    dataset = dataset.map(spectrogram.trim)


def build_train_dataset():
    folders = [x[0] for x in os.walk(TRAIN_PATH)][1:]
    print(folders)
    paths_set = stem_paths(folders)
    print(paths_set)
    dataset = tf.data.Dataset.from_tensor_slices(paths_set)
    dataset = dataset.shuffle(
        10000, seed=110, reshuffle_each_iteration=True
    )
    print(dataset.cardinality().numpy())
    dataset = dataset.map(lambda x: tf.py_function(load_multi_audio, [x, RESAMPLES], Tout=[tf.float32]))
    print("Samples mapped")
    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    print(dataset.cardinality().numpy())
    dataset = dataset.map(multi_wave_to_spec)
    dataset = dataset.map(spectrogram.trim_freq)
    dataset = dataset.map(spectrogram.trim)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
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