import tensorflow as tf
import numpy as np
import models
import datetime
import random
import argparse


parser = argparse.ArgumentParser(description='ResNet Hyper Parameters')
parser.add_argument('--layers', '-l', type=int,  required=True, help='Number of layers in the ResNet.')
parser.add_argument('--se-block', '-s', action='store_true', default=False, help='Use SE-block.')
parser.add_argument('--ratios', '-r', type=int, nargs='+', default=[16, 16, 16], help='Reduction ratios for each SE-block')
parser.add_argument('--regularization', '-r', type=float,  default=0.001, help='Amount of L2 regularization used in the ResNet')

args = parser.parse_args()


# "Tensorflow code will transparently run on a single GPU with no code changes."
#  https: // www.tensorflow.org/guide/gpu
# "Use the following line to confirm that TensorFlow is using the GPU."
GPUs = tf.config.experimental.list_physical_devices('GPU')
if GPUs:
    print('\nYou are using the following GPUs: {}.\n'.format(GPUs))
else:
    print('\nYou are currently not using any GPUs.\n')


L_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 32
LAMBDA = args.regularization
NUM_LAYERS = args.layers
SE_BLOCKS = args.se_block
RATIOS = args.ratios
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = "logs/" + current_time

def load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = tf.one_hot(y_train[:, 0], depth=10)
    y_test = tf.one_hot(y_test[:, 0], depth=10)
    return (x_train, y_train), (x_test, y_test)

def update_step(model, x, y, loss_func, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_func(y, y_pred)

    acc = accuracy(y, y_pred)
    trainable_vars = model.trainable_variables
    grad = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grad, trainable_vars))

    return loss.numpy(), acc.numpy()

def accuracy(labels, predictions):
    pred = tf.argmax(predictions, axis=1)
    labels = tf.argmax(labels, axis=1)
    m = tf.keras.metrics.Accuracy()
    _ = m.update_state(labels, pred)
    acc = m.result()
    return acc

def main():
    summery_writer = tf.summary.create_file_writer(LOG_DIR)

    print_every_n = 50

    (x_train, y_train), (x_test, y_test) = load_and_preprocess()

    n_train = x_train.shape[0]
    l2_regularizer = tf.keras.regularizers.l2(LAMBDA)

    model = models.ResNet(depth=NUM_LAYERS, se_block=SE_BLOCKS, ratios=RATIOS, regularizer=l2_regularizer)
    crossentropy = tf.keras.losses.CategoricalCrossentropy()
    adam = tf.keras.optimizers.Adam(learning_rate=L_RATE)

    for epoch in range(EPOCHS):
        loss = 0
        acc = 0
        for i in range(int(np.ceil(n_train//BATCH_SIZE))):
            x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            batch_loss, batch_acc = update_step(model, x_batch, y_batch, loss_func=crossentropy, optimizer=adam)
            loss += batch_loss
            acc += batch_acc

            if i % print_every_n == 0:
                if i != 0:
                    loss /= print_every_n
                    acc /= print_every_n
                print("Epoch {} Iteration {}/{} Loss {} Acc {}".format(epoch, i, n_train//BATCH_SIZE, loss, acc))
                loss = 0

        y_pred_test = model(x_test, training=False) # TODO: fixa så att man kan testa hela test set
        test_acc = accuracy(y_test, y_pred_test)
        test_loss = crossentropy(y_test, y_pred_test)

        print("Epoch {}, accuracy: {} loss {}".format(epoch, test_acc, test_loss))

        model.save_weights(filepath=LOG_DIR+"/{}ResNet_{}_weights/".format("SE_" if SE_BLOCKS else "", NUM_LAYERS))
        with summery_writer.as_default():
            tf.summary.scalar('Test Accuracy', test_acc, step=epoch)
            tf.summary.scalar('Test Loss', test_loss, step=epoch)

if __name__ == "__main__":
    main()
