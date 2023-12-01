import tensorflow as tf
import numpy as np
import soundfile as psf
import matplotlib.pyplot as plt
import data

SEPARATE_PATH = "/mnt/e/Downloads/Mystery.wav"
MODEL_PATH = "/mnt/e/Workspace/AISMentoring/f23/Seppit/seppit/Test_80.keras"

def SISNR_LOSS(target, est):
        s_target = target - tf.reduce_mean(target, keepdims=True)
        s_est = est - tf.reduce_mean(est, keepdims=True)

        mult = tf.reduce_sum(s_target * s_est, keepdims = True)

        s_tar = tf.reduce_sum(s_target ** 2, keepdims = True)+EPS
        proj = mult * s_target / s_tar

        e_noise = s_est - proj
        SDR = tf.reduce_sum(proj ** 2 ) / (tf.reduce_sum(e_noise ** 2) + EPS)

        si_snr = 10 * (tf.math.log(SDR + EPS) / tf.math.log(10.0))
 
        return -1.0 * si_snr #Greater magnitude of SDR is a good thing, so we need to make output negative   

def load_to_sep(path):
    model = tf.keras.models.load_model(MODEL_PATH)
    audio, _ = psf.read(path, dtype = "float32")
    audio_tensor = tf.convert_to_tensor(audio)
    stft_tensor = tf.transpose(
        tf.signal.stft(
            tf.transpose(audio_tensor),
            2048,
            512,
            window_fn=lambda f, dtype: tf.signal.hann_window(
                f, periodic=True, dtype=audio_tensor.dtype
            )
        ),
        perm=[1,2,0]
    )
    stft_tensor = tf.image.resize(stft_tensor, [tf.shape(stft_tensor)[1]/2, 512])
    padded_len = tf.shape(stft_tensor)[1] + (512 - tf.shape(stft_tensor)[1] % 512)
    num_splits = padded_len/512
    stft_tensor = tf.pad(tf.reduce_sum(stft_tensor,1), [[0,0],[0,padded_len-tf.shape(stft_tensor)],[0,0]])
    divided_spect = tf.split(stft_tensor, [512]*num_splits, axis = 1)
    prediction_spect = 3

def plot_prediction(path):
    dataset = data.build_train_dataset()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    for i, e in enumerate(dataset.take(5)):
        tmp = e[0].numpy()[0][:,:]
        plt.imshow(tmp)
        plt.savefig("testpicture" + str(i) + ".png")
        tmp = e[1].numpy()[0][:,:]
        plt.imshow(tmp)
        plt.savefig("testpicture_i" + str(i) + ".png")
        tmp = model.predict(e[0][0])
        plt.imshow(tmp)
        plt.savefig("testpicture_j" + str(i), ".png")





if __name__ == "__main__":
    plot_prediction("test")
