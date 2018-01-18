import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan import GAN

X_DIM = 784
Z_DIM = 100
HIDDEN_DIM = 128

BATCH_SIZE = 64
MAX_STEP = 100000


def sample_Z(n_sample, z_dim):
    return np.random.uniform(-1., 1., size=[n_sample, z_dim])


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


def train():
    model = GAN(X_DIM, Z_DIM, HIDDEN_DIM)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    i = 0

    for step in range(MAX_STEP):
        if step % 1000 == 0:
            z_sample = sample_Z(16, Z_DIM)
            g_sample = model.generate(sess, feed_dict={model.z: z_sample})
            fig = plot(g_sample)
            plt.savefig('results/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        x_batch, _ = mnist.train.next_batch(BATCH_SIZE)
        z_batch = sample_Z(BATCH_SIZE, Z_DIM)

        d_loss, g_loss = model.train(sess,
                                     d_feed_dict={
                                         model.x: x_batch,
                                         model.z: z_batch
                                     },
                                     g_feed_dict={
                                         model.z: z_batch
                                     })

        if step % 1000 == 0:
            print('step = {}'.format(step))
            print('d_loss = {}'.format(d_loss))
            print('g_loss = {}'.format(g_loss))


def setup():
    if not os.path.exists('results/'):
        os.mkdir('results/')


if __name__ == '__main__':
    setup()
    train()