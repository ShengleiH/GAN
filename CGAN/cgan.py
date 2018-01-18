import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class CGAN:
    def __init__(self, x_dim, z_dim, y_dim, hidden_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, x_dim), name='x')
        self.z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim), name='z')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, y_dim), name='y')

        '''
        Generator parameters
        '''
        self.G_W1 = tf.Variable(xavier_init([z_dim + y_dim, hidden_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))

        self.G_W2 = tf.Variable(xavier_init([hidden_dim, x_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[x_dim]))
        self.G_var_list = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

        '''
        Discriminator parameters
        '''
        self.D_W1 = tf.Variable(xavier_init([x_dim + y_dim, hidden_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))

        self.D_W2 = tf.Variable(xavier_init([hidden_dim, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.D_var_list = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

        '''
        Generator Discriminator beat each other
        '''
        # generator produce a fake image
        self.G_input = tf.concat([self.z, self.y], axis=1)
        self.G_H1 = tf.nn.relu(tf.matmul(self.G_input, self.G_W1) + self.G_b1)
        self.G_Sample = tf.nn.sigmoid(tf.matmul(self.G_H1, self.G_W2) + self.G_b2)

        # discriminator learns to know what is real image
        self.D_pos_input = tf.concat([self.x, self.y], axis=1)
        self.D_pos_H1 = tf.nn.relu(tf.matmul(self.D_pos_input, self.D_W1) + self.D_b1)
        self.D_pos_prob = tf.nn.sigmoid(tf.matmul(self.D_pos_H1, self.D_W2) + self.D_b2)

        # discriminator tries to distinguish the fake image
        self.D_neg_input = tf.concat([self.G_Sample, self.y], axis=1)
        self.D_neg_H1 = tf.nn.relu(tf.matmul(self.D_neg_input, self.D_W1) + self.D_b1)
        self.D_neg_prob = tf.nn.sigmoid(tf.matmul(self.D_neg_H1, self.D_W2) + self.D_b2)

        # generator and discriminator tries to measure how good they are
        self.D_loss = -tf.reduce_mean(tf.log(self.D_pos_prob) + tf.log(1. - self.D_neg_prob))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_neg_prob))

        # generator and discriminator tries to update themselves
        self.D_optimizer = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.D_var_list)
        self.G_optimizer = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.G_var_list)

    def train(self, sess, d_feed_dict, g_feed_dict):
        d_loss, _ = sess.run([self.D_loss, self.D_optimizer], feed_dict=d_feed_dict)
        g_loss, _ = sess.run([self.G_loss, self.G_optimizer], feed_dict=g_feed_dict)
        return d_loss, g_loss

    def generate(self, sess, feed_dict):
        g_sample = sess.run(self.G_Sample, feed_dict)
        return g_sample
