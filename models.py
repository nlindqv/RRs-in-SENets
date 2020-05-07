import tensorflow as tf


class ResNet(tf.keras.Model):
    '''
    Trying to replicate resnet_v1 from https://keras.io/examples/cifar10_resnet/
    '''
    def __init__(self, filters=(16, 32, 64), kernel_sizes=(3, 3, 3), ratios=(16, 16, 16), depth=20, se_block=False, regularizer=None):
        super(ResNet, self).__init__()
        # Here we define/initialize the layers with weights
        self.se_block = se_block
        self.main_layers = []

        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 ')
            # Start model definition.
        self.num_res_blocks = int((depth - 2) / 6)

        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            kernel_initializer="he_normal",
                                            padding="same",
                                            kernel_regularizer=regularizer)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        if self.se_block:
            self.se_block1 = SE_block(ratio=ratios[0], channels=16)

        # stage 1 layers
        for layer in range(self.num_res_blocks):
            self.main_layers.append(RES_block(n_filter=filters[0],
                                              kernel_size=kernel_sizes[0],
                                              se_block=self.se_block,
                                              se_ratio=ratios[0],
                                              regularizer=regularizer))

        # stage 2 layers
        self.main_layers.append(RES_block(n_filter=filters[1],
                                          kernel_size=kernel_sizes[1],
                                          first_block=True,
                                          se_block=self.se_block,
                                          se_ratio=ratios[1],
                                          regularizer=regularizer))
        for layer in range(self.num_res_blocks-1):
            self.main_layers.append(RES_block(n_filter=filters[1],
                                              kernel_size=kernel_sizes[1],
                                              se_block=self.se_block,
                                              se_ratio=ratios[1],
                                              regularizer=regularizer))

        # stage 3 layers
        self.main_layers.append(RES_block(n_filter=filters[2],
                                          kernel_size=kernel_sizes[2],
                                          first_block=True,
                                          se_block=self.se_block,
                                          se_ratio=ratios[2],
                                          regularizer=regularizer))
        for layer in range(self.num_res_blocks-1):
            self.main_layers.append(RES_block(n_filter=filters[2],
                                              kernel_size=kernel_sizes[2],
                                              se_block=self.se_block,
                                              se_ratio=ratios[2],
                                              regularizer=regularizer))

        # classification layers
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=8)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, kernel_regularizer=regularizer)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        # Here we do the feed forward computations
        current = self.conv1(inputs)
        current = self.batch_norm1(current, training=training)
        current = tf.keras.activations.relu(current)
        if self.se_block:
            current = self.se_block1(current)

        for layer in self.main_layers:
            current = layer(current, training=training)

        current = self.pool(current)
        current = self.flatten(current)
        current = self.dense(current)
        output = tf.keras.activations.softmax(current)
        return output

class SE_block(tf.keras.layers.Layer):
    def __init__(self, ratio, channels):
        super(SE_block, self).__init__()

        self.dense1 = tf.keras.layers.Dense(channels // ratio, activation='relu')
        self.dense2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=False, mask=None):
        current = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        current = self.dense1(current)
        current = self.dense2(current)
        residual = tf.transpose(inputs, (1, 2, 0, 3))
        output = tf.multiply(residual, current)
        output = tf.transpose(output, (2, 0, 1, 3))
        return output

class RES_block(tf.keras.layers.Layer):
    def __init__(self, n_filter, kernel_size, first_block=False, se_ratio=16, se_block=False, regularizer=None):
        super(RES_block, self).__init__()
        self.first_block = first_block

        self.cnn_block = CNN_block(filters=n_filter,
                                   kernal_size=kernel_size,
                                   first_block=first_block,
                                   se_ratio=se_ratio,
                                   se_block=se_block,
                                   regularizer=regularizer)
        if self.first_block: # if it is the first layer of a stage, then the residual needs to change dims, which is done here.
            self.residual_layer = tf.keras.layers.Conv2D(filters=n_filter,
                                                         kernel_size=1,
                                                         strides=2,
                                                         activation=None,
                                                         kernel_regularizer=regularizer)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        current = self.cnn_block(inputs, training=training)
        if self.first_block:
            residual = self.residual_layer(inputs)
        else:
            residual = inputs

        output = tf.keras.activations.relu(current + residual)
        return output

class CNN_block(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernal_size=3, se_block=False, se_ratio=16, first_block=False, regularizer=None):
        super(CNN_block, self).__init__()
        self.se_block = se_block
        if first_block:
            conv1_strides = 2
        else:
            conv1_strides = 1

        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(kernal_size, kernal_size),
                                            padding='same',
                                            strides=conv1_strides,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=regularizer)
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(kernal_size, kernal_size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=regularizer)

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        if self.se_block:
            self.se_block1 = SE_block(ratio=se_ratio, channels=filters)

    @tf.function
    def call(self, inputs, training=False, mask=None):
        current = self.conv1(inputs)
        current = self.batch_norm1(current, training=training)
        current = tf.keras.activations.relu(current)

        current = self.conv2(current)
        current = self.batch_norm2(current, training=training)
        if self.se_block:
            current = self.se_block1(current)

        return current


