import tensorflow as tf
import models
import main
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



weight_path = "logs/20200509-143023/ResNet_8_weights/checkpoint"

def get_confusion_matrix(weight_path, data):
    (x_test, y_test) = data
    x_test = x_test[:500]
    y_test = y_test[:500]

    model = models.ResNet(depth=8)
    _ = model(x_test[0:2])
    model.load_weights(weight_path)

    predictions = model(x_test, training=False).numpy()
    predictions = tf.argmax(predictions, axis=1)
    labels = tf.argmax(y_test, axis=1)
    confusion_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=10, dtype=tf.float32)
    print(confusion_matrix)
    confusion_matrix = tf.transpose(tf.divide(tf.transpose(confusion_matrix), tf.reduce_sum(y_test, axis=0)))
    print(tf.reduce_sum(y_test, axis=0))
    return confusion_matrix.numpy()


(_, _), data = main.load_dataset()

matrix = get_confusion_matrix(weight_path=weight_path, data=data)
df_cm = pd.DataFrame(matrix, range(10), range(10))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}) # font size

plt.show()

