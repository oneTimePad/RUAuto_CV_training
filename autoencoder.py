import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os



n_inputs = 28*28
n_hidden1 = 150
noise_level = 1.0

lr = 0.01
n_epoch =10
batch_size = 150

X = tf.placeholder(tf.float32, shape=(None,n_inputs),name="X")
#X_noisy = X + noise_level * tf.random_normal(tf.shape(X))
X_noisy = X
with tf.name_scope('network'):
    h1 = slim.fully_connected(X_noisy,150,activation_fn=tf.nn.relu)
    outputs = slim.fully_connected(h1,28*28,activation_fn=None)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(outputs-X))
with tf.name_scope('train'):
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.minimize(loss)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

if not os.path.exists('auto/'):
    os.makedirs('auto/')

i = 0
init = tf.global_variables_initializer()
mnist = input_data.read_data_sets("./MNIST")
n_test_digits = 10
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={X:X_batch})
        print(sess.run(loss,feed_dict={X:X_batch}))
        if epoch % 2 == 0:
            X_test = mnist.test.images[:n_test_digits]
            samples = sess.run(outputs,feed_dict={X:X_test})
            fig = plot(samples)
            plt.savefig('auto/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
