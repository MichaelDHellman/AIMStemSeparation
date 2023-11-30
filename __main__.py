from ossaudiodev import SNDCTL_DSP_SETFRAGMENT
from pickletools import optimize
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
from model import unet
import data
import psutil

EPS = 1e-10
PATH = "/mnt/e/Workspace/AISMentoring/F23/Seppit/MUSDB/test/Al James - Schoolboy Facination/mixture.wav"

#class SISNR_LOSS(tf.keras.losses.Loss):
#    def __init__(self):
#        super().__init__()

class UN_Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print((psutil.virtual_memory().used)/1024/1024)
        self.model.save("test_" + str(epoch) + ".keras")
    
    def on_train_batch_end(self, batch, logs=None):
         print("Starting batch: " + str(batch))

def SISNR(target, est):
     s_t = tf.transpose(est) * target



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


if __name__ =="__main__":
    model = unet.unet((1024,512,1))
    dataset = data.build_train_dataset()
    callback = UN_Callback()
    for e in dataset.take(1):
        print(SISNR_LOSS(e[1], e[0]))
    model.compile(optimizer='adam',loss=SISNR_LOSS)
    model.fit(dataset, epochs=20, steps_per_epoch=100,callbacks=[callback])


    