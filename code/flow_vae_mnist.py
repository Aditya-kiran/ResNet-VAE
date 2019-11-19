from flows import PlanarFlow, ResnetFlow
from losses import elbo_loss, vanilla_vae_loss, elbo_loss_resnet, cross_entropy_loss
from tb_logger import Logger
from utils import copy_files

import argparse
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from IPython import embed

config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement = True
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1

parser = argparse.ArgumentParser()
parser.add_argument( '--flow', type=str, default='None', help='resnet or planar')
parser.add_argument( '--mode', type=str, default='train', help='resnet or planar')
parser.add_argument('--load', action='store_true', help='loads model')
parser.add_argument( '--log-dir', type=str, default='logs/flow_vae', help='Logging directory - logs to tensorboard and saves code')
parser.add_argument( '--exp-name', type=str, default='temp', help='name of the current experiment')

args = parser.parse_args()

if args.flow == 'planar':
	print('--------------------------------------------------------')
	print('Executing Planar Normalizing Flow VAE')
	print('--------------------------------------------------------')
	FLOW = 'planar'
elif args.flow == 'resnet':
	print('--------------------------------------------------------')
	print('Executing ResNet Flow VAE')
	print('--------------------------------------------------------')
	FLOW = 'resnet'
else:
	print('--------------------------------------------------------')
	print('Executing Vanilla VAE')
	print('--------------------------------------------------------')
	FLOW = None

####################################
#Flow paramteres
if FLOW == 'resnet':
	num_flows = 8
	num_cells_per_flow = 4
	dt = tf.constant(1.0/(num_cells_per_flow*num_flows), name='dt')
	approximation = 2 # 1 , 2 , 3 or 4 - None for exact calculation of the determinant of jacobian
####################################
#Normalizing Flow paramteres
if FLOW == 'planar':
	num_flows = 10
####################################

logdir = os.path.join(args.log_dir, str(FLOW), args.exp_name)

##########################
#Saves the current version of the code in the log directory
copy_files(logdir) # saves the python files in the logging directory so that I dont forget what parameters worked the best!            
##########################

EPOCHS = 100
learning_rate = 0.001
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
x_train = mnist.train.images
x_test = mnist.test.images
x_dim = x_train.shape[1]
y_test= mnist.test.labels
batch_size = 100
z_dim = 2
hidden = 64
num_batches = x_train.shape[0]//batch_size

def sample_z_tf(mean, log_var):
	with tf.variable_scope('reparameterization_trick'):
		epsilon = tf.random_normal(shape = tf.shape(mean)) 
		z_sampled = mean + tf.exp(log_var / 2) * epsilon
	return z_sampled

def sample_z_np(mean, var):
	with tf.variable_scope('Sampling_z'):
		epsilon =np.random.normal(0,1)
		z_sampled_np =mean + np.exp(var / 2) * epsilon 
	return z_sampled_np

def encoder(X): 
	with tf.variable_scope('Encoder'):
		with tf.variable_scope('Encoder_parameters'):
			we1 = tf.get_variable('we1', shape = (x_dim,hidden))
			be1 = tf.get_variable('be1', shape=(hidden), initializer=tf.zeros_initializer())
			we2 = tf.get_variable('we2', shape = (hidden,hidden))
			be2 = tf.get_variable('be2', shape=(hidden), initializer=tf.zeros_initializer())
			w_mean = tf.get_variable('w_mean', shape=[hidden,z_dim])
			b_mean = tf.get_variable('b_mean', shape = [z_dim], initializer=tf.zeros_initializer())
			w_variance = tf.get_variable('w_variance', shape = [hidden,z_dim])
			b_variance = tf.get_variable('b_variance', shape = [z_dim], initializer=tf.zeros_initializer())
		with tf.variable_scope('hidden1'):
			h = tf.nn.relu(tf.matmul(X, we1) + be1)
		with tf.variable_scope('hidden2'):
			h = tf.nn.relu(tf.matmul(h, we2) + be2)
		with tf.variable_scope('mean'):
			z_mean = tf.matmul(h, w_mean) + b_mean
		with tf.variable_scope('variance'):
			z_var = tf.matmul(h, w_variance) + b_variance   
	return z_mean, z_var, h

def decoder(z):
	with tf.variable_scope('Decoder'):
		with tf.variable_scope('Decoder_parameters', reuse=tf.AUTO_REUSE):
			w_d1 = tf.get_variable(name='wd1', shape=[z_dim,hidden])
			b_d1 = tf.get_variable(name='bd1', shape=[hidden], initializer=tf.zeros_initializer())
			w_d2 = tf.get_variable(name='wd2', shape=[hidden,hidden])
			b_d2 = tf.get_variable(name='bd2', shape=[hidden], initializer=tf.zeros_initializer())
			w_d3 = tf.get_variable(name='wd3', shape=[hidden,x_dim])
			b_d3 = tf.get_variable(name='bd3', shape=[x_dim], initializer=tf.zeros_initializer())

		with tf.variable_scope('hidden1'):   
			h = tf.nn.relu(tf.matmul(z,w_d1) + b_d1)
		with tf.variable_scope('hidden2'):   
			h = tf.nn.relu(tf.matmul(h,w_d2) + b_d2)
		with tf.variable_scope('reconstructed_image'):  
			logits = tf.matmul(h,w_d3) + b_d3
			output = tf.nn.sigmoid(logits)  
	return output, logits

def sample_image(z, flow=None):
	if flow:
		with tf.variable_scope("Flow", reuse=tf.AUTO_REUSE):
			z_k, *_ = flow.flow(z)
	else:
		z_k = z
	output,_ = decoder(z_k)
	return output

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

X = tf.placeholder(tf.float32, shape =[None, x_dim], name='input_images')
z = tf.placeholder(tf.float32, shape =[None, z_dim], name='input_images')

z_mean, log_var, h = encoder(X)
z0 = sample_z_tf(z_mean,log_var)
z_var = tf.exp(log_var)
z_k = z0


flow = None
if FLOW:
	with tf.variable_scope('Flow'):
		if FLOW == 'resnet':
			flow = ResnetFlow(num_flows, num_cells_per_flow, z_dim, dt, approximation=approximation)
			z_k, sum_log_detj, _ = flow.flow(z0)
			z0_regenerated = flow.inverse(z_k)
		if FLOW == 'planar':
			flow = PlanarFlow(z_dim, num_flows)
			z_k, sum_log_detj = flow.flow(z0)
			z0_regenerated = flow.inverse(z_k)

prob,_ = decoder(z_k)
X_samples = sample_image(z, flow)
global_step = tf.Variable(0, trainable=False, name='global_step')

if FLOW == 'planar':
	loss_op = elbo_loss(X, prob,
						z_mu=z_mean, z_var= z_var, z0= z0, 
						zk= z_k, logdet_jacobian= sum_log_detj)
elif FLOW == 'resnet':
	loss_op = elbo_loss_resnet(X, prob, 
						z_mu=z_mean, z_var= z_var, z0= z0, 
						zk= z_k, logdet_jacobian= sum_log_detj)
else:
	loss_op = vanilla_vae_loss(prob, X, z_mean, z_var)

nll = tf.reduce_mean(cross_entropy_loss(prob, X))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=global_step)

tb_logger = Logger(log_dir=logdir, name='')

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

loss_train = []
loss_test = []

saver = tf.train.Saver(max_to_keep=10)

if args.load:
	saver.restore(sess, logdir+'/model_latest.ckpt')
	print("Model restored.")

if args.mode == 'train':
	for epoch in range(EPOCHS):
		loss=[]
		for i in range(num_batches):
			x_mini_batch, _ = mnist.train.next_batch(batch_size)
			_, loss, nll_loss = sess.run([train_op, loss_op, nll], feed_dict={X: x_mini_batch})
			step = tf.train.global_step(sess, tf.train.get_global_step())

			tb_logger.scalar_summary('train/nll_per_step', nll_loss, step)
			tb_logger.scalar_summary('train/neg_elbo_per_step', loss, step)

		tb_logger.scalar_summary('train/nll_per_epoch', nll_loss, epoch)
		tb_logger.scalar_summary('train/neg_elbo_per_epoch', loss, epoch)
		print('[Epoch: {} --- Loss: {}]'.format(epoch, loss))

		test_elbo_loss, test_nll_loss = sess.run([loss_op, nll], feed_dict={X: x_test})
		print('nll loss on the test set: {}'.format(test_nll_loss))
		print('elbo loss on the test set: {}'.format(test_elbo_loss))
		tb_logger.scalar_summary('test/nll_per_epoch', test_nll_loss, epoch)
		tb_logger.scalar_summary('test/neg_elbo_per_epoch', test_elbo_loss, epoch)

		if epoch%10==0:
			save_path = saver.save(sess, logdir+'/model_'+str(epoch)+'.ckpt')
			print('Saved model in: ' + save_path)
			samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

			fig = plot(samples)
			if not os.path.exists(logdir+'/out/'):
				os.makedirs(logdir+'/out/')
			plt.savefig(logdir+'/out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
			plt.close(fig)

	save_path = saver.save(sess, logdir+'/model_latest.ckpt')
	print('Saved model in: ' + save_path)

elif args.mode == 'test':
	elbo_loss, nll_loss = sess.run([loss_op, nll], feed_dict={X: x_test})
	print('nll_loss on the test set: {}'.format(nll_loss))
	print('elbo_loss on the test set: {}'.format(elbo_loss))

	plt.figure(figsize=(10, 10))
	n = 10
	im_size = 28
	figure = np.zeros((im_size * n, im_size * n))

	for j in range(n):
		for i in range(n):
			z_sample = np.random.normal(size=z_dim).reshape(1, z_dim)

			x_decoded = sess.run(X_samples, feed_dict={z: z_sample})
			digit = x_decoded.reshape(im_size, im_size, 1)      
			d_x = i * im_size
			d_y = j * im_size
			figure[d_x:d_x + im_size, d_y:d_y + im_size] = digit[:, :, 0]

	plt.imshow(figure, cmap='Greys_r')
	plt.show()
	embed()


















