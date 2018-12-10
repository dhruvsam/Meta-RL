# Include Libraries
import numpy as np
import tensorflow as tf
import gym
import tensorflow.contrib.layers as layers
# from mujoco_py import load_model_from_xml, MjSim, MjViewer
import policies
import value_functions as vfuncs
import utils_pg as utils

# Environment setup
#env = "CartPole-v0"
env="InvertedPendulum-v1"

#env = "HalfCheetah-v2"

# discrete = isinstance(env.action_space, gym.spaces.Discrete)
# observation_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n if discrete else env.action_space.shape[0]
max_ep_len = 1000
num_traj = 10
#traj_length = max_ep_len*(observation_dim + 2)
latent_size = 25
use_baseline = True



# Feed forward network (multi-layer-perceptron, or mlp)

def build_mlp(mlp_input,output_size,scope,n_layers,size,output_activation=None):
    '''
    Build a feed forward network
    '''
    Input = mlp_input
    with tf.variable_scope(scope):
        # Dense Layers
        for i in range(n_layers-1):
            dense = tf.layers.dense(inputs = Input, units = size, activation = tf.nn.relu, bias_initializer=tf.constant_initializer(1.0))
            Input = dense
        # Fully Connected Layer
        out = layers.fully_connected(inputs = Input, num_outputs = output_size, activation_fn=output_activation)
    return out

class MetaLearner():
    def __init__(self, env, max_ep_len, num_traj,latent_size ):
        self.env = gym.make(env)
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        self.max_ep_len = max_ep_len
        self.num_traj = num_traj
        self.traj_length = self.max_ep_len*(self.observation_dim + 1 + self.env.action_space.shape[0]) # TO Change
        self.use_baseline = True
        self.latent_size = latent_size
        self.feature_size = self.observation_dim + 1 + self.env.action_space.shape[0]
        self.lr = 3e-2
        self.num_layers = 1
        self.layers_size = 16
        self.num_mdps = 10
        self.model = self.env.env.model
        self.sim = MjSim(self.model)

        # build model
        self.ConstructGraph()

    def add_placeholders(self):
        self.observation_placeholder_explore = tf.placeholder(tf.float32, shape=(None,self.observation_dim))
        if(self.discrete):
            self.action_placeholder_explore = tf.placeholder(tf.int32, shape=(None))
            self.action_placeholder_exploit = tf.placeholder(tf.int32, shape=(None))
        else:
            self.action_placeholder_explore = tf.placeholder(tf.float32, shape=(None,self.action_dim))
            self.action_placeholder_exploit= tf.placeholder(tf.float32, shape=(None,self.action_dim))

        self.baseline_target_placeholder = tf.placeholder(tf.float32, shape= None)
        self.advantage_placeholder_explore = tf.placeholder(tf.float32, shape=(None))

        #self.encoder_input_placeholder = tf.placeholder(tf.float32, shape= (self.num_traj,self.traj_length))
        self.encoder_input_placeholder = tf.placeholder(tf.float32, [None, None, self.feature_size])
        self.decoder_input_placeholder = tf.placeholder(tf.float32, shape= (1,self.latent_size))
        self.sequence_length_placeholder = tf.placeholder(tf.int32, [None, ])

        self.observation_placeholder_exploit = tf.placeholder(tf.float32, shape=(None,self.observation_dim))
        #TODO
        self.advantage_placeholder_exploit = tf.placeholder(tf.float32, shape=(None))


    def build_policy_explore(self, scope = "policy_explore"):
        if (self.discrete):
            self.action_logits = build_mlp(self.observation_placeholder_explore,self.action_dim,scope = scope,n_layers=self.num_layers,size = self.layers_size,output_activation=None)
            self.explore_action = tf.multinomial(self.action_logits,1)
            self.explore_action = tf.squeeze(self.explore_action, axis=1)
            self.explore_logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.action_logits, labels = self.action_placeholder_explore)

        else:
            action_means = build_mlp(self.observation_placeholder_explore,self.action_dim,scope,n_layers=self.num_layers, size = self.layers_size,output_activation=None)
            init = tf.constant(np.random.rand(1, 2))
            log_std = tf.get_variable("log_std", [self.action_dim])
            self.explore_action =   action_means + tf.multiply(tf.exp(log_std),tf.random_normal(shape = (self.action_dim,1),mean=0,stddev=1))
            mvn = tf.contrib.distributions.MultivariateNormalDiag(action_means, tf.exp(log_std))
            self.explore_logprob =  mvn.log_prob(value = self.action_placeholder_explore, name='log_prob')



    def build_policy_exploit(self, scope = "policy_exploit"):
        if(self.discrete):
            #self.exploit_action_logits = (tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(self.observation_placeholder_exploit,self.d_W1) + self.d_B1), self.d_W2) + self.d_B2),self.d_W3) + self.d_B3)
            self.exploit_action_logits = tf.matmul(self.observation_placeholder_exploit,self.d_W3) + self.d_B3
            self.exploit_action = tf.multinomial(self.exploit_action_logits,1)
            self.exploit_action = tf.squeeze(self.exploit_action, axis=1)
            self.exploit_logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.exploit_action_logits, labels = self.action_placeholder_exploit)
        else:
            action_means = tf.matmul(self.observation_placeholder_exploit,self.d_W3) + self.d_B3
            init = tf.constant(np.random.rand(1, 2))
            log_std = tf.get_variable("exploit_log_prob", [self.action_dim])
            self.exploit_action =   action_means + tf.multiply(tf.exp(log_std),tf.random_normal(shape = (self.action_dim,1),mean=0,stddev=1))
            mvn = tf.contrib.distributions.MultivariateNormalDiag(action_means, tf.exp(log_std))
            self.exploit_logprob =  mvn.log_prob(value = self.action_placeholder_exploit, name='exploit_log_prob')

        #self.loss_grads_exploit = self.exploit_logprob * self.advantage_placeholder_exploit

    def NNEncoder(self, scope = "NNEncoder"):
        self.Z = build_mlp(self.encoder_input_placeholder,self.latent_size,scope = scope,n_layers=3,size = 60,output_activation=None)



    # input [num_traj, length, features (obs + action + reward) ]
    def LSTMEncoder(self, scope = "LSTMEncoder"):
        self.hidden_size = 64
        initializer = tf.random_uniform_initializer(-1, 1)
        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, self.feature_size, initializer=initializer)
        cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, self.latent_size)
        self.output, _ = tf.nn.dynamic_rnn(cell_out,self.encoder_input_placeholder,self.sequence_length_placeholder,dtype=tf.float32,)
        batch_size = tf.shape(self.output)[0]
        max_length = tf.shape(self.output)[1]
        out_size = int(self.output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (self.sequence_length_placeholder - 1)
        flat = tf.reshape(self.output, [-1, out_size])
        self.Z = tf.reduce_mean(tf.gather(flat, index), axis = 0)


    def Decoder(self, decoder_out_dim, scope):
        return build_mlp(self.decoder_input_placeholder,decoder_out_dim,scope = scope,n_layers=1,size = 16,output_activation=None)


    def sample_paths_explore(self, env,mdp_num,Test = False, num_episodes = None):
        paths = []
        self.length = []
        episode_rewards = []
        self.sim.model.opt.gravity[-1] = -5 - mdp_num
        env.env.gravity = 5 + 2*mdp_num
        for i in range(self.num_traj):
            pad = False
            state = env.reset()
            new_states,states,new_actions, actions, rewards = [], [], [], [], []
            episode_reward = 0
            for step in range(self.max_ep_len):
                if (pad):
                    states.append([0]*self.observation_dim)
                    if(self.discrete):
                        actions.append(0)
                    else:
                        actions.append([0]*self.action_dim)
                    rewards.append(0)
                else:
                    states.append(state)
                    new_states.append(state)
                    action = self.sess.run(self.explore_action, feed_dict={self.observation_placeholder_explore : states[-1][None]})[0]
                    state, reward, done, info = env.step(action)
                    actions.append(action)
                    new_actions.append(action)
                    rewards.append(reward)
                    episode_reward += reward
                    if (done or step == self.max_ep_len-1):
                        episode_rewards.append(episode_reward)
                        self.length.append(step + 1)
                        pad = True

            #print("explore",np.array(actions))
            path = {"new_obs" : np.array(new_states),"observation" : np.array(states),
                                "reward" : np.array(rewards),"new_acs" : np.array(new_actions),
                                "action" : np.array(actions)}
            paths.append(path)
        return paths, episode_rewards

    def sample_paths_exploit(self, env,Z,mdp_num,Test = False, num_episodes = None):
        self.sim.model.opt.gravity[-1] = -5 - mdp_num
        paths = []
        num = 0
        episode_rewards = []
        for i in range(self.num_traj):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0
            for step in range(self.max_ep_len):
                states.append(state)
                action = self.sess.run(self.exploit_action, feed_dict={self.observation_placeholder_exploit : state[None], self.decoder_input_placeholder: Z})[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                if (done or step == self.max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break
            #print("exploit",np.array(actions))
            path = {"observation" : np.array(states),
                                "reward" : np.array(rewards),
                                "action" : np.array(actions)}
            paths.append(path)

        #print("exploit success: ", num)
        return paths, episode_rewards

    def get_returns(self,paths, explore = False):
        all_returns = []
        m = 0
        for path in paths:
            rewards = path["reward"]
            returns = []
            if(explore):
                length = self.length[m]
            else:
                length = len(rewards)
            #print(self.length[m])
            for i in range(length):
                path_returns = 0
                k = 0
                for j in range(i,length):
                    path_returns = path_returns + rewards[j]*(1)**k
                    k = k+1
                returns.append(path_returns)
            all_returns.append(returns)
            m+=1
        returns = np.concatenate(all_returns)
        return returns

    def stack_trajectories(self,paths):
        trajectories = []
        for path in paths:
            rewards = path["reward"]
            states = path["observation"]
            action = path["action"]
            SAR = []
            for i in range(len(states)):
                SAR.append(list(states[i]) + list(action[i]) + [rewards[i]])
            trajectories.append(SAR)

        return np.array(trajectories)

    def addBaseline(self):
        self.baseline = build_mlp(self.observation_placeholder_explore,1,scope = "baseline",n_layers=self.num_layers, size = self.layers_size,output_activation=None)
        self.baseline_loss = tf.losses.mean_squared_error(self.baseline_target_placeholder,self.baseline,scope = "baseline")
        baseline_adam_optimizer =  tf.train.AdamOptimizer(learning_rate = self.lr)
        self.update_baseline_op = baseline_adam_optimizer.minimize(self.baseline_loss)

    def calculate_advantage(self,returns, observations):
        if (self.use_baseline):
            baseline = self.sess.run(self.baseline, {input_placeholder:observations})
            adv = returns - baseline
            adv = (adv - np.mean(adv))/np.std(adv)
        else:
            adv = returns
        return adv



    def ConstructGraph(self):
        tf.reset_default_graph()

        self.add_placeholders()

        self.build_policy_explore()
        self.explore_policy_loss = -tf.reduce_sum(self.explore_logprob * self.advantage_placeholder_explore)
        self.loss_grads_explore = -self.explore_logprob * self.advantage_placeholder_explore
        self.tvars_explore = tf.trainable_variables()
        self.gradients_explore = tf.gradients(self.explore_policy_loss,self.tvars_explore)

        #self.addBaseline()

        self.baseline = build_mlp(self.observation_placeholder_explore,1,scope = "baseline",n_layers=1, size = 16,output_activation=None)
        self.baseline_loss = tf.losses.mean_squared_error(self.baseline_target_placeholder,self.baseline,scope = "baseline")
        baseline_adam_optimizer =  tf.train.AdamOptimizer(learning_rate = self.lr)
        self.update_baseline_op = baseline_adam_optimizer.minimize(self.baseline_loss)

        #Encoder LSTM
        self.LSTMEncoder()

        self.decoder_len = 16
        #decoder weights
        #self.d_W1 = self.Decoder(scope = "W1", decoder_out_dim = self.observation_dim*self.decoder_len)
        #self.d_W2 = self.Decoder(scope = "W2", decoder_out_dim = self.decoder_len*self.decoder_len)
        #self.d_W3 = self.Decoder(scope = "W3", decoder_out_dim = self.decoder_len*action_dim)
        self.d_W3 = self.Decoder(scope = "W3", decoder_out_dim = self.observation_dim*self.action_dim)

        #self.d_W1 = ((self.d_W1 - (tf.reduce_max(self.d_W1) + tf.reduce_min(self.d_W1))/2)/(tf.reduce_max(self.d_W1) - tf.reduce_min(self.d_W1)))*2
        #self.d_W2 = ((self.d_W2 - (tf.reduce_max(self.d_W2) + tf.reduce_min(self.d_W2))/2)/(tf.reduce_max(self.d_W2) - tf.reduce_min(self.d_W2)))*2
        #self.d_W3 = ((self.d_W3 - (tf.reduce_max(self.d_W3) + tf.reduce_min(self.d_W3))/2)/(tf.reduce_max(self.d_W3) - tf.reduce_min(self.d_W3)))*2

        #self.d_W1 = tf.reshape(self.d_W1, [self.observation_dim, self.decoder_len])
        #self.d_W2 = tf.reshape(self.d_W2, [self.decoder_len, self.decoder_len])
        #self.d_W3 = tf.reshape(self.d_W3, [self.decoder_len, self.action_dim])
        self.d_W3 = tf.reshape(self.d_W3, [self.observation_dim, self.action_dim])

        # decoder output bias
        #self.d_B1 = tf.reshape(self.Decoder(decoder_out_dim = self.decoder_len, scope = "B1"), [self.decoder_len])
        #self.d_B2 = tf.reshape(self.Decoder(decoder_out_dim = self.decoder_len, scope = "B2"), [self.decoder_len])
        self.d_B3 = tf.reshape(self.Decoder(decoder_out_dim = self.action_dim, scope = "B3"), [self.action_dim])
        #self.d_B1 = ((self.d_B1 - (tf.reduce_max(self.d_B1) + tf.reduce_min(self.d_B1))/2)/(tf.reduce_max(self.d_B1) - tf.reduce_min(self.d_B1)))*2
        #self.d_B2 = ((self.d_B2 - (tf.reduce_max(self.d_B2) + tf.reduce_min(self.d_B2))/2)/(tf.reduce_max(self.d_B2) - tf.reduce_min(self.d_B2)))*2


        # exploit policy
        self.build_policy_exploit()
        #self.d = [self.d_W1, self.d_B1, self.d_W2, self.d_B2, self.d_W3, self.d_B3]
        self.exploit_policy_loss = -tf.reduce_sum(self.exploit_logprob * self.advantage_placeholder_exploit)
        self.d = [self.d_W3, self.d_B3]
        self.gradients_exploit = tf.gradients(self.exploit_policy_loss,self.d)

        # train encoder and decoder
        adam_optimizer_exploit =  tf.train.AdamOptimizer(self.lr*0.3)
        self.output_train_op = adam_optimizer_exploit.minimize(self.exploit_policy_loss)
        # train original network
        adam_optimizer_explore = tf.train.AdamOptimizer(self.lr)
        self.input_train_op = adam_optimizer_explore.minimize(self.explore_policy_loss)

    def initialize(self):
        self.ConstructGraph()
        # create tf session
        self.sess = tf.Session()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train_step(self):
        # sample num_traj*num_MDPs
        for i in range(self.num_mdps):
            explore_paths, explore_rewards = self.sample_paths_explore(self.env,i)
            self.scores_eval_explore = self.scores_eval_explore + explore_rewards
            observations_explore = np.concatenate([path["observation"] for path in explore_paths])
            new_observations_explore = np.concatenate([path["new_obs"] for path in explore_paths])
            new_actions_explore = np.concatenate([path["new_acs"] for path in explore_paths])
            actions_explore = np.concatenate([path["action"] for path in explore_paths])
            rewards_explore = np.concatenate([path["reward"] for path in explore_paths])
            returns_explore = self.get_returns(explore_paths, explore = True)
            #print(returns_explore)
            print("average reward explore: MDP", i, np.sum(explore_rewards)/num_traj, len(explore_rewards))

#             baseline_explore = self.sess.run(self.baseline, {self.observation_placeholder_explore:new_observations_explore})
#             adv = returns_explore - np.squeeze(baseline_explore)
#             advantages_explore = (adv - np.mean(adv))/np.std(adv)


#             # update the baseline

#             self.sess.run(self.update_baseline_op, {self.observation_placeholder_explore:new_observations_explore,
#                                            self.baseline_target_placeholder : returns_explore})

            # calculate explore gradients
    #         grads_explore = self.sess.run(self.gradients_explore, feed_dict={
    #                     self.observation_placeholder_explore : observations_explore,
    #                     self.action_placeholder_explore : actions_explore,
    #                     self.advantage_placeholder_explore : returns_explore})
            #print("explore",grads_explore )
            # form trajectory matrix

            M = np.array(self.stack_trajectories(explore_paths))


            #encoder LSTM
            Z = self.sess.run(self.Z, feed_dict = {self.encoder_input_placeholder: M,self.sequence_length_placeholder: self.length })
            Z = np.reshape(Z,[1,len(Z)])
            #print("Z",Z)
            #print(self.sess.run(self.d, feed_dict = {self.decoder_input_placeholder: Z}))
            #print("d",self.d)
            # sample paths
            tvars = tf.trainable_variables()
            tvars_vals = self.sess.run(tvars[-5:-1])
            #print(tvars_vals)
            for j in range(2):
                exploit_paths, exploit_rewards = self.sample_paths_exploit(self.env,Z,i)
                # get observations, actions and rewards
                observations_exploit = np.concatenate([path["observation"] for path in exploit_paths])
                actions_exploit = np.concatenate([path["action"] for path in exploit_paths])
                rewards_exploit = np.concatenate([path["reward"] for path in exploit_paths])
                returns_exploit = self.get_returns(exploit_paths)
                print("average reward exploit: MDP", i, np.sum(exploit_rewards) / num_traj, len(exploit_rewards))
                self.scores_eval_exploit = self.scores_eval_exploit + exploit_rewards


            # exploit grads
    #         grads_exploit = self.sess.run(self.gradients_exploit,feed_dict={
    #                     self.observation_placeholder_exploit : observations_exploit,
    #                     self.action_placeholder_exploit : actions_exploit,
    #                     self.advantage_placeholder_exploit : returns_exploit,
    #                     self.decoder_input_placeholder: Z})

            #train encoder and decoder network
                self.sess.run(self.output_train_op, feed_dict={
                                self.observation_placeholder_exploit : observations_exploit,
                                self.action_placeholder_exploit : actions_exploit,
                                self.advantage_placeholder_exploit : returns_exploit,
                                self.decoder_input_placeholder: Z})

            # find advantage for input network
    #         advantage_explore = 0
    #         for i in range(len(grads_exploit)):
    #             l1 = grads_exploit[i]
    #             l2 = grads_explore[i]
    #             advantage_explore = advantage_explore + np.matmul(l1.flatten(), l2.flatten())

            # train input policy
            self.sess.run(self.input_train_op, feed_dict={
                            self.observation_placeholder_explore : new_observations_explore,
                            self.action_placeholder_explore : new_actions_explore,
                            self.advantage_placeholder_explore : returns_explore})

    def test(self):
        explore_paths, explore_rewards = self.sample_paths_explore(self.env, Test = True)
        M = self.stack_trajectories(explore_paths)
        Z = self.sess.run(self.Z, feed_dict = {self.encoder_input_placeholder: M,self.sequence_length_placeholder: self.length })
        Z = np.reshape(Z,[1,len(Z)])
        # sample paths
        exploit_paths, exploit_rewards = self.sample_paths_exploit(self.env,Z, Test = True)
        print("average reward exploit", np.sum(exploit_rewards) / num_traj, len(exploit_rewards))

    def train(self):
        with tf.device("/gpu:0") as dev:
            self.initialize()
            num_epochs = 200
            self.scores_eval_explore = []
            self.scores_eval_exploit = []
            for epoch in range(num_epochs):
                print("epoch number: ", epoch)
                self.train_step()

a = MetaLearner(env, max_ep_len, num_traj, latent_size)
a.train()
