import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

n_epochs = 100000
code = 100
batch_size = 128
n_hidden = 128
ng_output=28*28
nd_output=1
lr = 0.01

Z = tf.placeholder(tf.float32,shape=(None,code),name='z')
X = tf.placeholder(tf.float32,shape=(None,ng_output),name='x')

g_layers =  [(n_hidden,tf.nn.relu),(ng_output,None)]
d_layers =  [(n_hidden,tf.nn.relu),(nd_output,None)]

#gen net
g = tf.nn.sigmoid(slim.stack(Z,slim.fully_connected,g_layers,scope='gen'))

#dis net on data
dx = slim.stack(X,slim.fully_connected,d_layers,scope='dis')
#dis net on gen data
dg = slim.stack(g,slim.fully_connected,d_layers,scope='dis',reuse=True)

with tf.name_scope('dis_train'):
    opt = tf.train.AdamOptimizer()
    dloss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dx,labels=tf.ones_like(dx)))
    dloss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dg,labels=tf.zeros_like(dg)))
    dloss = dloss_r+dloss_f

    dtrain_op = opt.minimize(dloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='dis'))
with tf.name_scope('gen_train'):
    opt = tf.train.AdamOptimizer()
    gloss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dg,labels=tf.ones_like(dg)))
    gtrain_op = opt.minimize(gloss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='gen'))



init= tf.global_variables_initializer()


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

if not os.path.exists('gan/'):
    os.makedirs('gan/')

def sample(batch_size,code):
    return np.random.uniform(-1.,1.,size=[batch_size,code])

mnist = input_data.read_data_sets("./MNIST")

#change
i = 0
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        #sample
        if epoch % 1000 == 0:
            samples = sess.run(g,feed_dict={Z:sample(16,code)})
            fig = plot(samples)
            plt.savefig('gan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
        #train discriminator
        for __ in range(1):
            X_batch, __ = mnist.train.next_batch(batch_size)
            _,dloss_val= sess.run([dtrain_op,dloss],feed_dict={X:X_batch,Z:sample(batch_size,code)})
        X_batch, __ = mnist.train.next_batch(batch_size)
        #train generator
        _,gloss_val= sess.run([gtrain_op,gloss],feed_dict={Z:sample(batch_size,code)})
        if epoch%1000 ==0:
            print("DLoss %f" %dloss_val)
            print("Gloss %f" %gloss_val)
