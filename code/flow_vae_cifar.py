from flows import NormalizingPlanarFlow, ResnetFlow
from losses import elbo_loss, vanilla_vae_loss, elbo_loss_resnet
from tb_logger import Logger
from utils import copy_files

import argparse
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


class CifarLoader(object):
	def __init__(self, source_files):
		self._source = source_files
		self._i = 0
		self.images = None
		self.labels = None

	def load(self):
		data = [unpickle(f) for f in self._source]
		images = np.vstack([d["data"] for d in data])
		n = len(images)
		self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
						  .astype(float) / 255
		self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
		return self

	def next_batch(self, batch_size):
		x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
		self._i = (self._i + batch_size) % len(self.images)
		return x, y

DATA_PATH = "./cifar-10-batches-py"

def unpickle(file):
	with open(os.path.join(DATA_PATH, file), 'rb') as fo:
		dict = pickle.load(fo)
	return dict
	
def one_hot(vec, vals=10):
	n = len(vec)
	out = np.zeros((n, vals))
	out[range(n), vec] = 1
	return out

class CifarDataManager(object):
	def __init__(self):
		self.train = CifarLoader(["data_batch_{}".format(i) 
		for i in range(1, 6)]).load()
		self.test = CifarLoader(["test_batch"]).load()

def display_cifar(images, size):
	n = len(images)
	plt.figure()
	plt.gca().set_axis_off()
	im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
	plt.imshow(im)
	plt.show()
	
CIFAR = CifarDataManager()
print "Number of train images: {}".format(len(CIFAR.train.images))
print "Number of train labels: {}".format(len(CIFAR.train.labels))
print "Number of test images: {}".format(len(CIFAR.test.images))
print "Number of test images: {}".format(len(CIFAR.test.labels))
images = CIFAR.train.images
# display_cifar(images, 10)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement = True
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1

parser = argparse.ArgumentParser()
parser.add_argument( '--flow', type=str, default='None', help='resnet or planar')
parser.add_argument( '--approx', type=int, default=1, help='approx for resnet')
parser.add_argument( '--mode', type=str, default='train', help='resnet or planar')
parser.add_argument('--load', action='store_true', help='loads model')
parser.add_argument( '--log_dir', type=str, default='logs/CIFAR/flow_vae', help='Logging directory - logs to tensorboard and saves code')
parser.add_argument( '--exp_name', type=str, default='temp', help='name of the current experiment')
parser.add_argument( '--model', type=str, default='latest', help='name of the current experiment')

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
	num_flows= 5
	dt = tf.constant(1.0/num_flows, name='dt')
	approximation = args.approx # 1 , 2 , 3 or 4 - None for exact calculation of the determinant of jacobian
####################################
#Normalizing Flow paramteres
if FLOW == 'planar':
	num_flows= 5
####################################

logdir = os.path.join(args.log_dir,	str(FLOW), args.exp_name)

##########################
#Saves the current version of the code in the log directory
copy_files(logdir) # saves the python files in the logging directory so that I dont forget what parameters worked the best!            
##########################

EPOCHS = 100
learning_rate = 0.001
x_train = CIFAR.train.images
x_test = CIFAR.test.images
x_dim = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
y_test= CIFAR.test.labels
batch_size = 256
z_dim = 256
hidden = 512
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
			we1 = tf.get_variable('we1', shape = (1024,hidden))
			be1 = tf.get_variable('be1', shape=(hidden), initializer=tf.zeros_initializer())
			w_mean = tf.get_variable('w_mean', shape=[hidden,z_dim])
			b_mean = tf.get_variable('b_mean', shape = [z_dim], initializer=tf.zeros_initializer())
			w_variance = tf.get_variable('w_variance', shape = [hidden,z_dim])
			b_variance = tf.get_variable('b_variance', shape = [z_dim], initializer=tf.zeros_initializer())
		
		conv1 = tf.layers.conv2d(
							X,
							filters=32,
							kernel_size=5,
							strides=(1, 1),
							padding='valid',
							data_format='channels_last',
							dilation_rate=(1, 1),
							activation=tf.nn.relu)
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
		conv2 = tf.layers.conv2d(
							pool1,
							filters=64,
							kernel_size=5,
							strides=(1, 1),
							padding='valid',
							data_format='channels_last',
							dilation_rate=(1, 1),
							activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
		y = tf.layers.Flatten()(pool2)
		with tf.variable_scope('hidden'):
			h = tf.nn.relu(tf.matmul(y, we1) + be1)
		with tf.variable_scope('mean'):
			z_mean = tf.matmul(h, w_mean) + b_mean
		with tf.variable_scope('variance'):
			z_var = tf.matmul(h, w_variance) + b_variance   
	return z_mean, z_var, h

def decoder(z):
	with tf.variable_scope('Decoder'):
		with tf.variable_scope('Decoder_parameters'):
			w_d1 = tf.get_variable(name='wd1', shape=[z_dim,hidden])
			b_d1 = tf.get_variable(name='bd1', shape=[hidden], initializer=tf.zeros_initializer())
			w_d2 = tf.get_variable(name='wd2', shape=[hidden,1024])
			b_d2 = tf.get_variable(name='bd2', shape=[1024], initializer=tf.zeros_initializer())
			tf.add_to_collection('w_d1',w_d1)
			tf.add_to_collection('w_d2',w_d2)
			tf.add_to_collection('b_d1',b_d1)
			tf.add_to_collection('b_d2',b_d2)
		with tf.variable_scope('hidden'):	
			
			h = tf.nn.relu(tf.matmul(z,w_d1) + b_d1)
			h = tf.nn.relu(tf.matmul(h,w_d2) + b_d2)

			h_reshaped = tf.reshape(h,(-1, 4, 4, 64))

		deconv1 = tf.layers.conv2d_transpose(h_reshaped,
									filters=32,
									kernel_size=6,
									strides=(3, 3),
									padding='valid',
									data_format='channels_last',
									activation=tf.nn.relu,)	

		output = tf.layers.conv2d_transpose(deconv1,
									filters=3,
									kernel_size=4,
									strides=(2, 2),
									padding='valid',
									data_format='channels_last',
									activation=tf.nn.sigmoid,)	
  
	return output

X = tf.placeholder(tf.float32, shape =[None, 32, 32, 3], name='input_images')

z_mean, log_var, h = encoder(X)
z0 = sample_z_tf(z_mean,log_var)
z_var = tf.exp(log_var)
z_k = z0

if FLOW:
	with tf.variable_scope('Flow'):
		if FLOW == 'resnet':
			flow = ResnetFlow(num_flows, z_dim, dt, approximation=approximation)
			z_k, sum_log_detj = flow.flow(z0)
		if FLOW == 'planar':
			flow = NormalizingPlanarFlow(z0, z_dim)
			z_k, sum_log_detj = flow.planar_flow(z0, h, H=hidden, K=num_flows, Z=z_dim)

		
prob = decoder(z_k)

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
	loss_op = vanilla_vae_loss(prob, X, z_mean, log_var)


train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op, global_step=global_step)

tb_logger = Logger(log_dir=logdir, name='')

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

loss_train = []
loss_test = []

saver = tf.train.Saver(max_to_keep=10)


if args.load:
	saver.restore(sess, logdir+'/model_'+args.model+ '.ckpt')
	print("Model restored.")

if args.mode == 'train':
	for epoch in range(EPOCHS):
		loss=[]
		for i in range(num_batches):
			x_mini_batch, _ = CIFAR.train.next_batch(batch_size)

			# x_mini_batch = x_mini_batch.reshape([batch_size,-1])
			_, loss= sess.run([train_op, loss_op], feed_dict={X: x_mini_batch})
			step = tf.train.global_step(sess, tf.train.get_global_step())

			tb_logger.scalar_summary('train/loss_per_step', loss, step)

		tb_logger.scalar_summary('train/loss_per_epoch', loss, epoch)
		print('[Epoch: {} --- Loss: {}]'.format(epoch,loss))
		if epoch%10==0:
			save_path = saver.save(sess, logdir+'/model_'+str(epoch)+'.ckpt')
			print('Saved model in: ' + save_path)
	save_path = saver.save(sess, logdir+'/model_'+args.model+ '.ckpt')
	print('Saved model in: ' + save_path)


elif args.mode == 'test':
	if FLOW == 'resnet':
		Z = tf.placeholder(tf.float32, shape = [None,z_dim], name='latent_code')
		with tf.variable_scope("Flow", reuse=tf.AUTO_REUSE):
			z_k,_ = flow.flow(Z)
		z_k = tf.reshape(z_k, (-1, 1, z_dim))
	else:
		Z = tf.placeholder(tf.float32, shape = [None,z_dim], name='latent_code')

	with tf.variable_scope("", reuse=tf.AUTO_REUSE):
		generator = decoder(Z)	
	# flow = ResnetFlow(num_flows, z_dim, dt, approximation=approximation)
	
	embed()

	import matplotlib.pyplot as plt
	n = 20
	figure = np.zeros((32 * n, 32 * n, 3))

	for j in range(n):
		for i in range(n):
			z_sample = np.random.normal(size=z_dim).reshape(1, z_dim)

			x_decoded = sess.run([generator], feed_dict={Z: z_sample})
			digit = x_decoded[0].reshape(32, 32, 3)
			
			d_x = i * 32
			d_y = j * 32
			figure[d_x:d_x + 32, d_y:d_y + 32] = digit
			
	plt.figure(figsize=(10, 10))
	plt.imshow(figure)
	plt.show()


