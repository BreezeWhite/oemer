import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LayerNormalization,
    Activation,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Add,
    Concatenate
)

def conv_block(input_tensor, channel, kernel_size, strides=(2, 2), dilation_rate=1, dropout_rate=0.4):
    """Convolutional encoder block of U-net.

    The block is a fully convolutional block. The encoder block does not downsample the input feature,
    and thus the output will have the same dimension as the input.
    """

    skip = input_tensor

    input_tensor = LayerNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(
        channel, kernel_size, strides=strides, dilation_rate=dilation_rate, padding="same"
    )(input_tensor)

    input_tensor = LayerNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(
        channel, kernel_size, strides=(1, 1), dilation_rate=dilation_rate, padding="same"
    )(input_tensor)

    if strides != (1, 1):
        skip = Conv2D(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = Add()([input_tensor, skip])

    return input_tensor


def transpose_conv_block(input_tensor, channel, kernel_size, strides=(2, 2), dropout_rate=0.4):
    skip = input_tensor

    input_tensor = LayerNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2D(channel, kernel_size, strides=(1, 1), padding="same")(input_tensor)

    input_tensor = LayerNormalization()(Activation("relu")(input_tensor))
    input_tensor = Dropout(dropout_rate)(input_tensor)
    input_tensor = Conv2DTranspose(channel, kernel_size, strides=strides, padding="same")(input_tensor)

    if strides != (1, 1):
        skip = Conv2DTranspose(channel, (1, 1), strides=strides, padding="same")(skip)
    input_tensor = Add()([input_tensor, skip])

    return input_tensor


def semantic_segmentation(win_size=256, multi_grid_layer_n=1, multi_grid_n=5, out_class=2, dropout=0.4):
    """Improved U-net model with Atrous Spatial Pyramid Pooling (ASPP) block."""
    input_score = Input(shape=(win_size, win_size, 3), name="input_score_48")
    en = Conv2D(2**7, (7, 7), strides=(1, 1), padding="same")(input_score)

    en_l1 = conv_block(en, 2**7, (3, 3), strides=(2, 2))
    en_l1 = conv_block(en_l1, 2**7, (3, 3), strides=(1, 1))

    en_l2 = conv_block(en_l1, 2**7, (3, 3), strides=(2, 2))
    en_l2 = conv_block(en_l2, 2**7, (3, 3), strides=(1, 1))
    en_l2 = conv_block(en_l2, 2**7, (3, 3), strides=(1, 1))

    en_l3 = conv_block(en_l2, 2**7, (3, 3), strides=(2, 2))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))
    en_l3 = conv_block(en_l3, 2**7, (3, 3), strides=(1, 1))

    en_l4 = conv_block(en_l3, 2**8, (3, 3), strides=(2, 2))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))
    en_l4 = conv_block(en_l4, 2**8, (3, 3), strides=(1, 1))

    feature = en_l4
    for _ in range(multi_grid_layer_n):
        feature = LayerNormalization()(Activation("relu")(feature))
        feature = Dropout(dropout)(feature)
        m = LayerNormalization()(Conv2D(2**9, (1, 1), strides=(1, 1), padding="same", activation="relu")(feature))
        multi_grid = m
        for ii in range(multi_grid_n):
            m = LayerNormalization()(
                Conv2D(2**9, (3, 3), strides=(1, 1), dilation_rate=2**ii, padding="same", activation="relu")(feature)
            )
            multi_grid = Concatenate()([multi_grid, m])
        multi_grid = Dropout(dropout)(multi_grid)
        feature = Conv2D(2**9, (1, 1), strides=(1, 1), padding="same")(multi_grid)

    feature = LayerNormalization()(Activation("relu")(feature))

    feature = Conv2D(2**8, (1, 1), strides=(1, 1), padding="same")(feature)
    feature = Add()([feature, en_l4])
    de_l1 = transpose_conv_block(feature, 2**7, (3, 3), strides=(2, 2))

    skip = de_l1
    de_l1 = LayerNormalization()(Activation("relu")(de_l1))
    de_l1 = Concatenate()([de_l1, LayerNormalization()(Activation("relu")(en_l3))])
    de_l1 = Dropout(dropout)(de_l1)
    de_l1 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l1)
    de_l1 = Add()([de_l1, skip])
    de_l2 = transpose_conv_block(de_l1, 2**7, (3, 3), strides=(2, 2))

    skip = de_l2
    de_l2 = LayerNormalization()(Activation("relu")(de_l2))
    de_l2 = Concatenate()([de_l2, LayerNormalization()(Activation("relu")(en_l2))])
    de_l2 = Dropout(dropout)(de_l2)
    de_l2 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l2)
    de_l2 = Add()([de_l2, skip])
    de_l3 = transpose_conv_block(de_l2, 2**7, (3, 3), strides=(2, 2))

    skip = de_l3
    de_l3 = LayerNormalization()(Activation("relu")(de_l3))
    de_l3 = Concatenate()([de_l3, LayerNormalization()(Activation("relu")(en_l1))])
    de_l3 = Dropout(dropout)(de_l3)
    de_l3 = Conv2D(2**7, (1, 1), strides=(1, 1), padding="same")(de_l3)
    de_l3 = Add()([de_l3, skip])
    de_l4 = transpose_conv_block(de_l3, 2**7, (3, 3), strides=(2, 2))

    de_l4 = LayerNormalization()(Activation("relu")(de_l4))
    de_l4 = Dropout(dropout)(de_l4)
    out = Conv2D(out_class, (1, 1), strides=(1, 1), activation='softmax', padding="same", name="prediction")(de_l4)

    return Model(inputs=input_score, outputs=out)


def my_conv_block(inp, kernels, kernel_size=(3, 3), strides=(1, 1)):
    inp = L.Conv2D(kernels, kernel_size, strides=strides, padding='same', dtype=tf.float32)(inp)
    out = L.Activation("relu")(L.LayerNormalization()(inp))
    out = L.SeparableConv2D(kernels, kernel_size, padding='same', dtype=tf.float32)(out)
    out = L.Activation("relu")(L.LayerNormalization()(out))
    out = L.Dropout(0.3)(out)
    out = L.Add()([inp, out])
    out = L.Activation("relu")(L.LayerNormalization()(out))
    return out


def my_conv_small_block(inp, kernels, kernel_size=(3, 3), strides=(1, 1)):
    inp = L.Conv2D(kernels, kernel_size, strides=strides, padding='same', dtype=tf.float32)(inp)
    out = L.Activation("relu")(L.LayerNormalization()(inp))
    out = L.Dropout(0.3)(out)
    out = L.Add()([inp, out])
    out = L.Activation("relu")(L.LayerNormalization()(out))
    return out


def my_trans_conv_block(inp, kernels, kernel_size=(3, 3), strides=(1, 1)):
    inp = L.Conv2DTranspose(kernels, kernel_size, strides=strides, padding='same', dtype=tf.float32)(inp)
    #out = L.Activation("relu")(L.LayerNormalization()(inp))
    out = L.Conv2D(kernels, kernel_size, padding='same', dtype=tf.float32)(inp)
    out = L.Activation("relu")(L.LayerNormalization()(out))
    out = L.Dropout(0.3)(out)
    out = L.Add()([inp, out])
    out = L.Activation("relu")(L.LayerNormalization()(out))
    return out


def u_net(win_size=288, out_class=3):
    inp = L.Input(shape=(win_size, win_size, 3))
    tensor = L.SeparableConv2D(128, (3, 3), activation="relu", padding='same')(inp)

    l1 = my_conv_small_block(tensor, 64, (3, 3), strides=(2, 2))
    l1 = my_conv_small_block(l1, 64, (3, 3))
    l1 = my_conv_small_block(l1, 64, (3, 3))

    skip = my_conv_small_block(l1, 128, (3, 3), strides=(2, 2))
    l2 = my_conv_small_block(skip, 128, (3, 3))
    l2 = my_conv_small_block(l2, 128, (3, 3))
    l2 = my_conv_small_block(l2, 128, (3, 3))
    l2 = my_conv_small_block(l2, 128, (3, 3))
    l2 = L.Concatenate()([skip, l2])

    l3 = my_conv_small_block(l2, 256, (3, 3))
    l3 = my_conv_small_block(l3, 256, (3, 3))
    l3 = my_conv_small_block(l3, 256, (3, 3))
    l3 = my_conv_small_block(l3, 256, (3, 3))
    l3 = my_conv_small_block(l3, 256, (3, 3))
    l3 = L.Concatenate()([l2, l3])

    bot = my_conv_small_block(l3, 256, (3, 3), strides=(2, 2))  # 32
    st1 = L.SeparableConv2D(256, (3, 3), padding='same', dtype=tf.float32)(bot)
    st1 = L.Activation("relu")(L.LayerNormalization()(st1))
    st2 = L.SeparableConv2D(256, (3, 3), dilation_rate=(2, 2), padding='same', dtype=tf.float32)(bot)
    st2 = L.Activation("relu")(L.LayerNormalization()(st2))
    st3 = L.SeparableConv2D(256, (3, 3), dilation_rate=(6, 6), padding='same', dtype=tf.float32)(bot)
    st3 = L.Activation("relu")(L.LayerNormalization()(st3))
    st4 = L.SeparableConv2D(256, (3, 3), dilation_rate=(12, 12), padding='same', dtype=tf.float32)(bot)
    st4 = L.Activation("relu")(L.LayerNormalization()(st4))
    st = L.Concatenate()([st1, st2, st3, st4])
    st = L.Conv2D(256, (1, 1), padding='same', dtype=tf.float32)(st)
    norm = L.Activation("relu")(L.LayerNormalization()(st))
    bot = my_trans_conv_block(norm, 256, (3, 3), strides=(2, 2))  # 64

    tl3 = L.Conv2D(128, (3, 3), padding='same', dtype=tf.float32)(bot)
    tl3 = L.Activation("relu")(L.LayerNormalization()(tl3))
    tl3 = L.Concatenate()([tl3, l3])
    tl3 = my_conv_small_block(tl3, 128, (3, 3))
    tl3 = my_trans_conv_block(tl3, 128, (3, 3))

    # Head 1
    tl2 = L.Conv2D(128, (3, 3), padding='same', dtype=tf.float32)(tl3)
    tl2 = L.Activation("relu")(L.LayerNormalization()(tl2))
    tl2 = L.Concatenate()([tl2, l2])
    tl2 = my_conv_small_block(tl2, 128, (3, 3))
    tl2 = my_trans_conv_block(tl2, 128, (3, 3), strides=(2, 2))  # 128

    tl1 = L.Conv2D(128, (3, 3), padding='same', dtype=tf.float32)(tl2)
    tl1 = L.Activation("relu")(L.LayerNormalization()(tl1))
    tl1 = L.Concatenate()([tl1, l1])
    tl1 = my_conv_small_block(tl1, 128, (3, 3))
    tl1 = my_trans_conv_block(tl1, 128, (3, 3), strides=(2, 2))  # 256

    out1 = L.Conv2D(out_class, (1, 1), activation='softmax', padding='same', dtype=tf.float32)(tl1)

    # Head 2
    # tl2 = L.Conv2D(256, (3, 3), padding='same', dtype=tf.float32)(bot)
    # tl2 = L.Activation("relu")(L.LayerNormalization()(tl2))
    # tl2 = L.Concatenate()([tl2, l2])
    # tl2 = my_trans_conv_block(tl2, 256, (3, 3), strides=(2, 2))  # 128

    # tl1 = L.Conv2D(128, (3, 3), padding='same', dtype=tf.float32)(tl2)
    # tl1 = L.Activation("relu")(L.LayerNormalization()(tl1))
    # tl1 = L.Concatenate()([tl1, l1])
    # tl1 = my_trans_conv_block(tl1, 128, (3, 3), strides=(2, 2))  # 256

    # out2 = L.Conv2D(out_class, (1, 1), activation='softmax', padding='same', dtype=tf.float32)(tl1)
    return tf.keras.Model(inputs=inp, outputs=out1)


if __name__ == "__main__":
    m = u_net()
