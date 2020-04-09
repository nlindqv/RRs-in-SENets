import tensorflow as tf
import numpy as np
L_RATE = 0.01
EPOCHS = 10
#N_BATCHES = 5
BATCH_SIZE = 100
tf.compat.v1.disable_v2_behavior()


def one_hot(y, depth):
    one_hot_y = np.zeros(shape=(y.size, depth))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = one_hot(y_train[:, 0], depth=10)
    y_test = one_hot(y_test[:, 0], depth=10)
    return (x_train, y_train), (x_test, y_test)

def cnn_model(inputs):

    current = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer="Orthogonal",
        padding="same",
        activation=tf.nn.relu,
    )(inputs)

    current = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(current)
    current = se_block(current, 32)

    current = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="Orthogonal",
        activation=tf.nn.relu,
    )(current)

    current = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(current)
    current = se_block(current, 64)

    current = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="Orthogonal",
        activation=tf.nn.relu,
    )(current)

    # Fully connected layers
    current = tf.keras.layers.Flatten()(current)
    current = tf.keras.layers.Dense(64, activation='relu')(current)
    output = tf.keras.layers.Dense(10)(current)

    return output

def se_block(residual, ch, ratio = 16):
    current = tf.keras.layers.GlobalAveragePooling2D()(residual)
    current = tf.keras.layers.Dense(ch//ratio, activation='relu')(current)
    current = tf.keras.layers.Dense(ch, activation='sigmoid')(current)
    residual = tf.transpose(residual, (1,2,0,3))
    output = tf.multiply(residual, current)
    output = tf.transpose(output, (2,0,1,3))

    return output


def train():
    pass


def main():

    inputs = tf.keras.backend.placeholder(shape=(None, 32, 32, 3))
    label = tf.keras.backend.placeholder(shape=(None, 10))

    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    n_train = x_train.shape[0]
    output = cnn_model(inputs)

    loss = tf.keras.losses.mean_squared_error(label, output)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=L_RATE).minimize(loss)

    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(EPOCHS):
            for i in range(int(np.ceil(n_train/BATCH_SIZE))):
                x_train_batch, y_train_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE], y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                sess.run(optimizer,
                         feed_dict={
                             inputs: x_train_batch,
                             label: y_train_batch,
                         })
                acc = sess.run(accuracy,
                               feed_dict={
                                   inputs: x_test[:1000],
                                   label: y_test[:1000],
                               })
                print("Epoch {}, Batch {}, Accuracy {}".format(epoch, i, acc))
            #print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            #print_stats(sess, batch_features, batch_labels, cost, accuracy)


main()
