import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def double_convolve(filters, x):
    x = layers.Conv2D(filters, 5, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(filters, 5, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    return x

def encode(x, filters):
    x = double_convolve(filters, x)
    p = layers.MaxPooling2D(strides=(2,2))(x)
    return x, p


def decode(x, y, filters):
    x = layers.Conv2DTranspose(filters, 5, strides=(2,2), padding="same")(x)
    x = layers.Concatenate(axis=3)([x,y])
    x=double_convolve(filters, x)
    return x

def unet(input_shape = (4096, 4096, 1)):
    inputs = layers.Input(shape=input_shape)
    f1, p1 = encode(inputs, 16)
    f2, p2 = encode(p1, 32)
    f3, p3 = encode(p2, 64)
    f4, p4 = encode(p3, 128)
    f5, p5 = encode(p4, 256)

    neck = double_convolve(512, p5)

    p6 = decode(neck, f5, 256)
    p6 = layers.Dropout(.35)(p6)
    p7 = decode(p6, f4, 128)
    p7 = layers.Dropout(.35)(p7)
    p8 = decode(p7, f3, 64)
    p8 = layers.Dropout(.35)(p8)
    p9 = decode(p8, f2, 32)
    p10 = decode(p9, f1, 16)
    probs = layers.Conv2D(2, (4,4), dilation_rate=(2,2), activation = "sigmoid", padding="same")(p10)
    out = layers.Multiply()([probs, inputs])

    model = tf.keras.models.Model(inputs, out, name = "unet")
    model.summary()
    return model


if __name__ == "__main__":
    m = unet()
    print(m.summary())