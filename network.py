import tensorflow as tf

middleLen = 120
target_lr = 0.001

with tf.name_scope('input'):
    observation = tf.placeholder(tf.float32, [None,2], 'observation')
    G = tf.placeholder(tf.float32, [None,1], 'target')
    action = tf.placeholder(tf.float32, [None,3], 'action')


with tf.name_scope('action_net'):
    w_a1 = tf.Variable(tf.truncated_normal([2,middleLen], stddev=0.1),
    name='w_a1')
    b_a1 = tf.Variable(tf.constant(0.1, shape=[middleLen]), name='b_a1')
    pa_a1 = tf.matmul(observation, w_a1) + b_a1
    a_a1 = tf.nn.relu(pa_a1, name='a_a1')

    w_a2 = tf.Variable(tf.truncated_normal([middleLen,3], stddev=1),
    name='w_a2')
    b_a2 = tf.Variable(tf.constant(0.1, shape=[3]), name='b_a2')
    pa_a2 = tf.matmul(a_a1, w_a2) + b_a2
    action_prob = tf.nn.softmax(pa_a2, name='action_prob')

    with tf.name_scope('gradient'):
        chosen_action_prob = tf.reduce_sum(tf.multiply(action_prob, action),
        axis=1, keep_dims=True)
        log_prob = tf.log(chosen_action_prob)
        J = tf.multiply(log_prob, G)
        loss = - tf.reduce_mean(J)
        train_action = tf.train.AdamOptimizer(learning_rate=target_lr)\
        .minimize(loss)
