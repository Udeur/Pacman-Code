import numpy as np
import tensorflow as tf

#Laye one and two eliminated
#se


class DQN:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet'
        self.sess = tf.Session()
        #height and 6 eliminated
        self.x = tf.placeholder('float', [None, 10, 5], name=self.network_name + '_x')
        self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t')
        #set to five since "stay" is an option
        self.actions = tf.placeholder("float", [None, 5], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')

        I = 50  #Number of Input Values (hier: 10*5)
        A = 30  #Number of Neurons first Layer
        B = 20  #Number of Neurons second Layer
        C = 5   #Number of Output Valaues


        #Netz so angepasst, wie gestern besprochen mit abnehmender Neuronenanzahl; Jetzt: Erst 30 Neuronen, dann 20 dann 5 Output Values

        layer_name = 'fc1'
        self.w1 = tf.Variable(tf.truncated_normal([I, A], stddev=0.01))
        self.x_r = tf.reshape(self.x,[-1, I])
        self.b1 = tf.Variable(tf.ones([A])/10)
        self.y1 = tf.nn.sigmoid(tf.matmul(self.x_r,self.w1 )+self.b1)


        layer_name = 'fc2'

        self.w2 = tf.Variable(tf.random_normal([A, B], stddev=0.01))
        self.b2 = tf.Variable(tf.ones([B])/10)
        self.y2 = tf.nn.relu(tf.matmul(self.y1, self.w2)+self.b2)

        layer_name = 'fc3'

        self.w3 = tf.Variable(tf.random_normal([B, C],stddev=0.01))
        self.b3 = tf.Variable(tf.ones([C])/10)
        self.y3 = tf.add(tf.matmul(self.y2, self.w3),self.b3)



        # Q,Cost,Optimizer line 65 argument tf.multiply has either input self.y2_sipmle or self.y3
        self.discount = tf.constant(self.params['discount'])
        self.yj = tf.add(self.rewards, tf.multiply(1.0 - self.terminals, tf.multiply(self.discount, self.q_t)))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y3, self.actions), reduction_indices=1)
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step',
                                           trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'], self.params['rms_decay'], 0.0,
                                                 self.params['rms_eps']).minimize(self.cost,
                                                                                  global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess, self.params['load_file'])

    def train(self, bat_s, bat_a, bat_t, bat_n, bat_r):
        feed_dict = {self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals: bat_t,
                     self.rewards: bat_r}
        q_t = self.sess.run(self.y3, feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict = {self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals: bat_t, self.rewards: bat_r}
        _, cnt, cost = self.sess.run([self.rmsprop, self.global_step, self.cost], feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self, filename):
        self.saver.save(self.sess, filename)