"""
Value functions, for now we apply them to policy gradients. This uses Python3
importing syntax.
"""

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import utils_pg as utils




class NnValueFunction(object):
    """ Estimates the baseline function for PGs via neural network. """

    def __init__(self,num_layers, size, session, ob_dim=None, n_epochs=20, stepsize=1e-3):
        """ The network gets constructed upon initialization so future calls to
        self.fit will remember this.

        Right now we assume a preprocessing which results ob_dim*2+1 dimensions,
        and we assume a fixed neural network architecture (input-50-50-1, fully
        connected with tanh nonlineariites), which we should probably change.
        The number of outputs is one, so that ypreds_n is the predicted vector
        of state values, to be compared against ytargs_n. Since ytargs_n is of
        shape (n,), we need to apply a "squeeze" on the final predictions, which
        would otherwise be of shape (n,1). Bleh.
        """
        # Value function V(s_t) (or b(s_t)), parameterized as a neural network.
        
        self.ob_no = tf.placeholder(shape=[None, ob_dim*2+1], name="nnvf_ob", dtype=tf.float32)
        Input = self.ob_no
        for i in range(num_layers-1):
            out = layers.fully_connected(Input,
                num_outputs=size,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh)
            Input = out
            
        self.ypreds_n = layers.fully_connected(Input,
                num_outputs=1,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=None)
        self.ypreds_n = tf.reshape(self.ypreds_n, [-1]) # (?,1) --> (?,). =)

        # Form the loss function, which is the simple (mean) L2 error.
        self.n_epochs = n_epochs
        self.lrate    = stepsize
        self.ytargs_n = tf.placeholder(shape=[None], name="nnvf_y", dtype=tf.float32)
        self.l2_error = tf.reduce_mean(tf.square(self.ypreds_n - self.ytargs_n))
        self.fit_op   = tf.train.AdamOptimizer(self.lrate).minimize(self.l2_error)
        self.sess     = session


    def fit(self, X, y, session=None):
        """
        Updates weights (self.coef) with design matrix X (i.e. observations) and
        targets (i.e. actual returns) y. For now, assume that by refitting,
        we'll be doing it several times (`n_epoch` times).
        """
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        Xp = self.preproc(X)
        for i in range(self.n_epochs):
            _,err = self.sess.run(
                    [self.fit_op, self.l2_error],
                    feed_dict={self.ob_no: Xp,
                               self.ytargs_n: y
                    })


    def predict(self, X):
        """
        Predicts returns from observations (i.e. environment states) X. I also
        think we need a session here. No need to expand dimensions, BTW! It's
        effectively already done for us elsewhere.
        """
        Xp = self.preproc(X)
        return self.sess.run(self.ypreds_n, feed_dict={self.ob_no:Xp})


    def preproc(self, X):
        """ Let's add this here to increase dimensionality. """
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)
