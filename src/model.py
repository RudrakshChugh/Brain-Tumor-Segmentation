import tensorflow as tf
from tensorflow.keras import layers, activations, Input, Model

def create_conv_block(input_tensor, num_filters):
    """Create a two-layer Conv2D block with batch normalization and ReLU."""
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def create_unet(input_shape, num_filters=16, dropout=0.1):
    """Build the main 2D U-Net model used for binary tumor segmentation."""
    c1 = create_conv_block(input_shape, num_filters * 1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(dropout)(p1)
    c2 = create_conv_block(p1, num_filters * 2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)
    c3 = create_conv_block(p2, num_filters * 4)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)
    c4 = create_conv_block(p3, num_filters * 8)
    u5 = tf.keras.layers.Convolution2DTranspose(num_filters*8, (3, 3), strides=(2, 2), padding='same')(c4)
    u5 = tf.keras.layers.concatenate([u5, c3])
    u5 = tf.keras.layers.Dropout(dropout)(u5)
    c5 = create_conv_block(u5, num_filters*4)
    u6 = tf.keras.layers.Convolution2DTranspose(num_filters*4, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c2])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = create_conv_block(u6, num_filters*2)
    u7 = tf.keras.layers.Convolution2DTranspose(num_filters*2, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c1])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = create_conv_block(u7, num_filters*1)
    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = tf.keras.Model(inputs = [input_shape], outputs = [output])
    return model 

def ResidualBlock(width):
    """Residual block used by the alternate U-Net implementation."""
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply

def DownBlock(width, block_depth):
    """Downsampling block for the alternate residual U-Net."""
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply

def UpBlock(width, block_depth):
    """Upsampling block for the alternate residual U-Net."""
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply

def create_unet2(imsize, widths, block_depth):
    """Build an alternate residual U-Net model."""
    input_image = Input(shape=imsize)
    x = layers.Conv2D(widths[0], kernel_size=1)(input_image)
    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])
    out = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(x)
    return Model(inputs=input_image, outputs=out, name="UNet")
