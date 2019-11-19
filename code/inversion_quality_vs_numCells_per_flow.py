from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

class ResnetFlow(object):
	def __init__(self, n_flows, z_dim, dt):
		self.num_flows = n_flows
		self.z_dim = z_dim
		self.dt = dt
		with tf.variable_scope('resnet',reuse=tf.AUTO_REUSE):
			self.w1 = tf.get_variable(name='weight1', shape=[self.z_dim, self.z_dim], initializer=tf.constant_initializer(.1))
			self.b1 = tf.get_variable(name='bias1', shape=[self.z_dim], initializer=tf.constant_initializer(.5))
			self.w2 = tf.get_variable(name='weight2', shape=[self.z_dim, self.z_dim], initializer=tf.constant_initializer(.1))
			self.b2 = tf.get_variable(name='bias2', shape=[self.z_dim], initializer=tf.constant_initializer(.5))

	def resnet_block(self, z_in):
		h = tf.nn.tanh(tf.matmul(z_in, self.w1 ) + self.b1)
		h = tf.matmul(h, self.w2 ) + self.b2
		return h
	
	def flow(self, z0):
		z_in = z0
		for i in range(self.num_flows):
			res = self.resnet_block(z_in)
			z_k = z_in + res*self.dt
			z_in = z_k
		return z_k

	def inverse(self, z_k): 
		for flow_index in np.arange(self.num_flows, 0, -1):
			res = self.resnet_block(z_k)
			z_k = z_k - res*self.dt
		return z_k


z_dim = 2
Y = tf.placeholder(tf.float32, shape =[None, z_dim])

sess = tf.Session()
sess.run(tf.global_variables_initializer())


n_trials = 50
z_samples = np.random.normal(0.0, 1.0, [1000, z_dim, n_trials])

plotting_data=[]#(num_flows, error)
for num_flows in np.arange(1,65,2):
	print(num_flows)
	dt = 1.0/num_flows
	flow = ResnetFlow(num_flows, z_dim, dt)
	Y = tf.placeholder(tf.float32, shape =[None, z_dim])
	z_k = flow.flow(Y)
	z0_regenerated = flow.inverse(z_k)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	errors=[]
	for i in range(n_trials):
		z_end, inverted_z = sess.run([z_k, z0_regenerated], feed_dict={Y: z_samples[:,:,i]})
		errors.append(np.mean(np.linalg.norm(z_samples[:,:,i] - inverted_z, axis=1)))
	plotting_data.append([num_flows,np.mean(errors)])

plotting_data = np.asarray(plotting_data)
df = pd.DataFrame (plotting_data)
filepath = 'inversion_quality.xlsx'
df.to_excel(filepath, index=False)

plt.figure()
plt.plot(plotting_data[:,0], plotting_data[:,1])
plt.grid()
plt.xlabel('Number of RNN blocks')
plt.ylabel('Average L2 inversion error ')

plt.savefig('inversion_quality',dpi=200)
plt.close()