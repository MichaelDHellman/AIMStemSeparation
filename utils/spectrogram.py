import tensorflow as tf

def waveform_to_spectrogram(waveform):
    stft_tensor = tf.transpose(
        tf.signal.stft(
            tf.transpose(waveform),
            2048,
            512,
            window_fn=lambda f, dtype: tf.signal.hann_window(
                f, periodic=True, dtype=waveform.dtype
            )
        ),
        perm=[1, 2, 0],
    )
    spect = tf.abs(stft_tensor)
    return spect

def gain_to_db(spectrogram):
    return (20.0*tf.math.log(tf.maximum(spectrogram, 10e-9))) / tf.math.log(10.0)

def db_to_gain(spectrogram):
    return tf.pow(10.0, (spectrogram/20.0))

#def change_speed_batched_random(spectrograms):


def change_speed(spectrogram, mag):
    time = tf.shape(spectrogram)[0]
    time_alt = tf.cast(tf.cast(time, tf.float32) * mag, tf.int32)
    
    adj_spect = tf.image.resize(spectrogram, [time_alt, tf.shape(spectrogram)[1]], method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(adj_spect, time, tf.shape(spectrogram)[1])

def trim_freq(spect):
    tmp = spect[:, 0:1024, 0:512, :]
    return tmp

def trim(frame):
    tmp = frame[:2, :, :, :]
    s_inp, s_out = tf.split(tmp, num_or_size_splits=2, axis=0)
    tup = (s_inp, s_out)
    return tup
