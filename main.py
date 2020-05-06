import tensorflow as tf
import numpy as np
import models
import datetime

L_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32
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
        y_pred = model(x)
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
    print_every_i = 20
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    #x_train, y_train = x_train[:2000], y_train[:2000]
    n_train = x_train.shape[0]
    model = models.ResNet(se_block=False)
    crossentropy = tf.keras.losses.CategoricalCrossentropy()
    adam = tf.keras.optimizers.Adam(learning_rate=L_RATE)
    for epoch in range(EPOCHS):
        loss = 0
        acc = 0
        for i in range(int(np.ceil(n_train//BATCH_SIZE))):
            x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if epoch == 0 and i == 0:
                tf.summary.trace_on(graph=True, profiler=True)

            batch_loss, batch_acc = update_step(model, x_batch, y_batch, loss_func=crossentropy, optimizer=adam)
            loss += batch_loss
            acc += batch_acc

            if i % print_every_i == 0:
                if i != 0:
                    loss /= print_every_i
                    acc /= print_every_i
                with summery_writer.as_default():
                    tf.summary.scalar('loss', loss, step=i)
                    if epoch == 0 and i == 0:
                        tf.summary.trace_export(
                            name="my_func_trace",
                            step=0,
                            profiler_outdir=LOG_DIR)
                print("Epoch {} Iteration {}/{} Loss {} Acc {}".format(epoch, i, n_train//BATCH_SIZE, loss, acc))
                loss = 0

        y_pred_test = model(x_test[:500])
        acc = accuracy(y_test[:500], y_pred_test)
        print("Epoch {}, accuracy: {}".format(epoch, acc))

main()
