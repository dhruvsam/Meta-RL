"""
For managing policies. This seems like a better way to organize things. For now,
we have **stochastic** policies. This assumes the Python3 way of calling
superclasses' init methods.
TODO figure out a good way to integrate deterministic policies, figure out how
to get a good configuration file (for neural nets), etc. Lots of fun! :)
TODO figure out how to make assertions that we're in continuous vs discrete
spaces.
TODO have a net specification which we can use instead of hard-coding networks
here.
"""

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import utils_pg as utils


class StochasticPolicy(object):

    def __init__(self, sess, ob_dim, ac_dim):
        """
        Initializes the neural network policy. Right now there isn't much here,
        but this is a flexible design pattern for future versions of the code.
        """
        self.sess = sess

    def sample_action(self, x):
        """ To be implemented in the subclass. """
        raise NotImplementedError


class GaussianPolicy(StochasticPolicy):
    """ A policy where the action is to be sampled based on sampling a Gaussian;
    this is for continuous control. """

    def __init__(self,num_layers, size ,sess, ob_dim, ac_dim):
        super().__init__(sess, ob_dim, ac_dim)

        # Placeholders for our inputs. Note that actions are floats.
        self.ob_no = tf.placeholder(shape=[None, ob_dim], name="obs", dtype=tf.float32)
        self.ac_na = tf.placeholder(shape=[None, ac_dim], name="act", dtype=tf.float32)
        self.adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.n     = tf.shape(self.ob_no)[0]

        # Special to the continuous case, the log std vector, it's a parameter.
        # Also, make batch versions so we get shape (n,a) (or (1,a)), not (a,).
        self.logstd_a     = tf.get_variable("logstd", [ac_dim], initializer=tf.zeros_initializer())
        self.oldlogstd_a  = tf.placeholder(name="oldlogstd", shape=[ac_dim], dtype=tf.float32)
        self.logstd_na    = tf.ones(shape=(self.n,ac_dim), dtype=tf.float32) * self.logstd_a
        self.oldlogstd_na = tf.ones(shape=(self.n,ac_dim), dtype=tf.float32) * self.oldlogstd_a

        # The policy network and the logits, which are the mean of a Gaussian.
        # Then don't forget to make an "old" version of that for KL divergences.
        Input = self.ob_no
        for i in range(num_layers-1):
            out = layers.fully_connected(Input,
                num_outputs=size,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.relu)
            Input = out


        self.mean_na = layers.fully_connected(Input,
                num_outputs=ac_dim,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=None)
        
        self.oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)

        # Diagonal Gaussian distribution for sampling actions and log probabilities.
        self.logprob_n  = utils.gauss_log_prob(mu=self.mean_na, logstd=self.logstd_na, x=self.ac_na)
        self.sampled_ac = (tf.random_normal(tf.shape(self.mean_na)) * tf.exp(self.logstd_na) + self.mean_na)[0]

        # Loss function that we'll differentiate to get the policy  gradient
        self.surr_loss = - tf.reduce_mean(self.logprob_n * self.adv_n)
        self.stepsize  = tf.placeholder(shape=[], dtype=tf.float32)
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr_loss)

        # KL divergence and entropy among Gaussian(s).
        self.kl  = tf.reduce_mean(utils.gauss_KL(self.mean_na, self.logstd_na, self.oldmean_na, self.oldlogstd_na))
        self.ent = 0.5 * ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(self.logstd_a)


    def sample_action(self, ob):
        return self.sess.run(self.sampled_ac, feed_dict={self.ob_no: ob[None]})


    def update_policy(self, ob_no, ac_n, std_adv_n, stepsize):
        """
        The input is the same for the discrete control case, except we return a
        single log standard deviation vector in addition to our logits. In this
        case, the logits are really the mean vector of Gaussians, which differs
        among components (observations) in the minbatch. We return the *old*
        ones since they are assigned, then `self.update_op` runs, which makes
        them outdated.
        """
        feed = {self.ob_no: ob_no,
                self.ac_na: ac_n,
                self.adv_n: std_adv_n,
                self.stepsize: stepsize}
        _, surr_loss, oldmean_na, oldlogstd_a = self.sess.run(
                [self.update_op, self.surr_loss, self.mean_na, self.logstd_a],
                feed_dict=feed)
        return surr_loss, oldmean_na, oldlogstd_a


    def kldiv_and_entropy(self, ob_no, oldmean_na, oldlogstd_a):
        """ Returning KL diverence and current entropy since they can re-use
        some of the computation. For the KL divergence, though, we reuqire the
        old mean *and* the old log standard deviation to fully characterize the
        set of probability distributions we had earlier, each conditioned on
        different states in the MDP. Then we take the *average* of these, etc.,
        similar to the discrete case.
        """
        feed = {self.ob_no: ob_no,
                self.oldmean_na: oldmean_na,
                self.oldlogstd_a: oldlogstd_a}
        return self.sess.run([self.kl, self.ent], feed_dict=feed)
