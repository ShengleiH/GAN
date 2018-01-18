import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class GAN:
    def __init__(self, x_dim, z_dim, hidden_dim):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, x_dim), name='x')
        self.z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim), name='z')

        '''
        generator parameters
        '''
        self.G_W1 = tf.Variable(xavier_init([z_dim, hidden_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))

        self.G_W2 = tf.Variable(xavier_init([hidden_dim, x_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[x_dim]))
        self.G_var_list = [self.G_W1, self.G_b1, self.G_W2, self.G_b2]

        '''
        discriminator paraments
        '''
        self.D_W1 = tf.Variable(xavier_init([x_dim, hidden_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))

        self.D_W2 = tf.Variable(xavier_init([hidden_dim, 1]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))
        self.D_var_list = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]


        '''
        generator and discriminator beat each other
        '''
        # generator produce a fake image
        self.G_h1 = tf.nn.relu(tf.matmul(self.z, self.G_W1) + self.G_b1)
        self.G_Sample = tf.nn.sigmoid(tf.matmul(self.G_h1, self.G_W2) + self.G_b2)

        # discriminator learns to know what is true image
        self.D_pos_h1 = tf.nn.relu(tf.matmul(self.x, self.D_W1) + self.D_b1)
        self.D_pos_prob = tf.sigmoid(tf.matmul(self.D_pos_h1, self.D_W2) + self.D_b2)

        # discriminator tries to distinguish fake image
        self.D_neg_h1 = tf.nn.relu(tf.matmul(self.G_Sample, self.D_W1) + self.D_b1)
        self.D_neg_prob = tf.nn.sigmoid(tf.matmul(self.D_neg_h1, self.D_W2) + self.D_b2)

        # generator and discriminator measures how good they are
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