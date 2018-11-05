
import numpy as np
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers



def build_mlp(mlp_input,output_size,scope,n_layers=config.n_layers,size=config.layer_size,output_activation=None):
    '''
    Build a feed forward network
    '''
    Input = mlp_input
    with tf.variable_scope(scope):
        # Dense Layers
        for i in range(n_layers-1):
            dense = tf.layers.dense(inputs = Input, num_outputs = size, activation = tf.nn.relu, bias_initializer=tf.constant_initializer(1.0))
            Input = dense
        # Fully Connected Layer
        out = layers.fully_connected(inputs = Input, num_outputs = output_size, activation_fn=output_activation)
    return out


class MetaRL:
    """
    Class that defines the initial exploratory Policy
    """
    def __init__(self, env, config):
        # discrete action space or continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph
        Set up the observation, action, and advantage placeholder
        """
        self.observation_placeholder = tf.placeholder(tf.float32, shape=(None,self.observation_dim))

        if self.discrete:
          self.action_placeholder = tf.placeholder(tf.int32, shape=(None))
        else:
          self.action_placeholder = tf.placeholder(tf.float32, shape=(None,self.action_dim))

        # Define a placeholder for advantages
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None))

        #TODO
        self.encoder_input_placeholder = tf.placeholder(tf.float32, shape=(None,self.Encoder_Input_dim))

    def build_policy_network_op(self, scope = "policy_network"):
        """
        builds the policy network
        """
        if self.discrete:
          action_logits = build_mlp(self.observation_placeholder,self.action_dim,scope = scope,n_layers=self.config.n_layers,size = self.config.layer_size,output_activation=None)
          self.sampled_action = tf.multinomial(action_logits,1)
          self.sampled_action = tf.squeeze(self.sampled_action, axis=1)
          self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits = action_logits, labels = self.action_placeholder)

        else:
          action_means = build_mlp(self.observation_placeholder,self.action_dim,scope,n_layers=self.config.n_layers,size = self.config.layer_size,output_activation=None)
          init = tf.constant(np.random.rand(1, 2))
          log_std = tf.get_variable("log_std", [self.action_dim])
          self.sampled_action =   action_means + tf.multiply(tf.exp(log_std),tf.random_normal(shape = (self.action_dim,1),mean=0,stddev=1))
          mvn = tf.contrib.distributions.MultivariateNormalDiag(action_means, tf.exp(log_std))

          self.logprob =  mvn.log_prob(value = self.action_placeholder, name='log_prob')

    def build(self):
        """
        Build model by adding all necessary variables

        """

        # add placeholders
        self.add_placeholders_op()
        # create policy net
        self.build_policy_network_op()



    def initialize(self):
        """
        Assumes the graph has been constructed (have called self.build())
        Creates a tf Session and run initializer of variables

        You don't have to change or use anything here.
        """
        # create tf session
        self.sess = tf.Session()
        # tensorboard stuff
        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def sample_paths(self, env, num_episodes = None):
        """
        Sample T trajectories
        """
        for i in range(self.batchsize):
            state = env.reset()
            states, actions, rewards = [], [], []
            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None]})[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                if (done):
                    break
            path = {"observation" : np.array(states),
                            "reward" : np.array(rewards),
                            "action" : np.array(actions)}
            paths.append(path)

        return paths

    def Construct_EncoderInput(self, paths):
        #TODO
        return Input

    def Encoder(self, scope = "encoder"):
        """
        Defines the encoder
        """
        output = build_mlp(self.encoder_input_placeholder,self.latent_1_dim,scope = scope,n_layers=self.config.encoder_layers,size = self.config.layer_size,output_activation=None)
        # TODO MAXPOOL ?
        Z = build_mlp(output,self.latent_2_dim,scope = scope,n_layers=self.config.encoder_layers,size = self.config.layer_size,output_activation=None)

        return Z

    def Decoder(self,Z,scope = "decoder"):
        d1 = build_mlp(Z,self.d1_out_dim,scope = scope,n_layers=self.config.decoder_layers,size = self.config.d1_layer_size,output_activation=None)
        d2 = build_mlp(Z,self.d2_out_dim,scope = scope,n_layers=self.config.decoder_layers,size = self.config.d2_layer_size,output_activation=None)
        d3 = build_mlp(Z,self.d3_out_dim,scope = scope,n_layers=self.config.decoder_layers,size = self.config.d3_layer_size,output_activation=None)

        return d1, d2, d3

    def PolicyExploit(self, Input, d1, d2, d3, scope = "exploit"):
        if self.discrete:
            action_logits = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(Input,d1)), d2)),d3))
            self.new_action = tf.multinomial(action_logits,1)
            self.new_action = tf.squeeze(self.sampled_action, axis=1)
            self.new_logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits = action_logits, labels = self.action_placeholder)
        else:
            action_means = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(Input,d1)), d2)),d3))
            log_std = tf.get_variable("log_std", [self.action_dim])
            self.new_action =   action_means + tf.multiply(tf.exp(log_std),tf.random_normal(shape = (self.action_dim,1),mean=0,stddev=1))
            mvn = tf.contrib.distributions.MultivariateNormalDiag(action_means, tf.exp(log_std))

            self.new_logprob =  mvn.log_prob(value = self.action_placeholder, name='log_prob')

    def loss_op(self):
        self.loss = -tf.reduce_sum(self.logprob * self.advantage_placeholder)

    def add_optimizer_op(self):
        adam_optimizer =  tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train_op = adam_optimizer.minimize(self.loss)


    #TODO - Change
    def add_baseline_op(self, scope = "baseline"):
        self.baseline = build_mlp(self.observation_placeholder,1,scope,n_layers=self.config.n_layers,size = self.config.layer_size,output_activation=None)
        self.baseline_target_placeholder = tf.placeholder(tf.float32, shape= None)
        loss = tf.losses.mean_squared_error(self.baseline_target_placeholder,self.baseline,scope = scope)
        adam_optimizer =  tf.train.AdamOptimizer(learning_rate = self.lr)
        self.update_baseline_op = adam_optimizer.minimize(loss)


    def initialize(self):
        # create tf session
        self.sess = tf.Session()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def get_returns(self, paths):
        all_returns = []
        for path in paths:
          rewards = path["reward"]
          returns = []
          for i in range(len(rewards)):
              path_returns = 0
              k = 0
              for j in range(i,len(rewards)):
                  path_returns = path_returns + rewards[j]*(config.gamma)**k
                  k = k+1
              returns.append(path_returns)
          all_returns.append(returns)
        returns = np.concatenate(all_returns)

    def calculate_advantage(self, returns, observations):
        adv = returns
        if self.config.use_baseline:
            baseline = self.sess.run(self.baseline, {self.observation_placeholder:observations})
            adv = returns - baseline

        if self.config.normalize_advantage:
            m = np.mean(adv)
            std_dev = np.std(adv)
            adv = (adv - m)/std_dev


    def update_baseline(self, returns, observations):
        self.sess.run(self.update_baseline_op, feed_dict={
                    self.observation_placeholder : observations,
                    self.baseline_target_placeholder : returns})

    def train(self):
        """
        trains the decoder and the encoder
        """
        scores_eval = []
        for t in range(self.config.num_batches):
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            scores_eval = scores_eval + total_rewards
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            if self.config.use_baseline:
                self.update_baseline(returns, observations)

            self.sess.run(self.train_op, feed_dict={
                          self.action_placeholder : actions,
                          self.advantage_placeholder : advantages})

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            print(msg)

    def evaluate(self, env=None, num_episodes=1):
        
