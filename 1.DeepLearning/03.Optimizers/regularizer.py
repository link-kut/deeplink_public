from functools import partial

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("~/git/deeplink_public/data/")

tf.reset_default_graph()

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
momentum = 0.25
scale = 0.001

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)


with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    my_dense_layer = partial(
        tf.layers.dense, activation=leaky_relu, kernel_initializer=he_init,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale)
    )

    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, name="outputs")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    print(reg_losses)

    base_loss = tf.reduce_mean(xentropy, name="base_loss")
    print(base_loss)

    loss = tf.add_n([base_loss] + reg_losses, name="loss")
    print(loss)

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


n_epochs = 30
batch_size = 50

if __name__ == "__main__":
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(mnist.test.labels)//batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={is_training: True, X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={is_training: False, X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={is_training: False, X: mnist.test.images, y: mnist.test.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
