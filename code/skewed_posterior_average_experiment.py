import numpy as np
import pandas as pd
import seaborn as sb
from scipy import stats

import pymc3 as pm
import theano.tensor as tt
from IPython import embed

data = pd.read_csv('cancermortality.csv')
x_data = np.array(data.y)
n_data = np.array(data.n)

############
# MCMC

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

mcmc_m_means = []
mcmc_k_means = []
mcmc_covs = []

planar_m_means = []
planar_k_means = []
planar_covs = []

ours_m_means = []
ours_k_means = []
ours_covs = []

# for _ in range(10):
# 	with pm.Model() as model:
# 		def logp(value):
# 			y1 = value[0]
# 			y2 = value[1] 
# 			#m = 1 / (1 + np.exp(-y1))  # 1 / (1 + exp(-x))
# 			#K = np.exp(y2) # K = tt.exp(y2)
# 			#return -np.log(m) -np.log(1-m) - 2.0*np.log(1+K)  # original version
# 			#return np.log( 1 + np.exp(-y1) ) - ( -y1 - np.log( 1 + np.exp(-y1) ) ) - 2.0*np.log(1+ np.exp(y2)) # simplified version
# 			return  2.0*tt.log( 1 + tt.exp(-y1) ) +y1 - 2.0*tt.log(1+ tt.exp(y2)) # simplified version

# 		y = pm.DensityDist('y', logp, shape=2, testval = [0,1])
# 		m = pm.Deterministic('m', pm.invlogit(y[0]))
# 		K = pm.Deterministic('K', pm.math.exp(y[1]))
# 		y_1 = pm.Deterministic('y_1', y[0])
# 		y_2 = pm.Deterministic('y_2', y[1])
		
# 		X0 = pm.BetaBinomial('X', alpha=K*m, beta=K*(1-m), n=n_data, observed=x_data)
# 		start = pm.find_MAP()
# 		trace = pm.sample(20000, step=pm.Metropolis(), 
# 						  njobs=1, chains=5, burn =5000, start=start)

# 		m1, m2 = trace['y_1'], trace['y_2']
# 		m=sigmoid(m1)
# 		K=np.exp(m2)
# 		print('MCMC: m_mean:', np.mean(m))
# 		mcmc_m_means.append(np.mean(m))
# 		print('MCMC: m_mean:', np.std(m))
# 		print('MCMC: K_mean:', np.mean(K))
# 		mcmc_k_means.append(np.mean(K))
# 		print('MCMC: K_mean:', np.std(K))
# 		print('MCMC: cov:', np.cov(m,K))
# 		mcmc_covs.append(np.cov(m,K))

# 	###########
# 	# Planar

# 	with pm.Model() as model:
# 		def logp(value):
# 			y1 = value[0]
# 			y2 = value[1] 
# 			#m = 1 / (1 + np.exp(-y1))  # 1 / (1 + exp(-x))
# 			#K = np.exp(y2) # K = tt.exp(y2)
# 			#return -np.log(m) -np.log(1-m) - 2.0*np.log(1+K)  # original version
# 			#return np.log( 1 + np.exp(-y1) ) - ( -y1 - np.log( 1 + np.exp(-y1) ) ) - 2.0*np.log(1+ np.exp(y2)) # simplified version
# 			return  2.0*tt.log( 1 + tt.exp(-y1) ) +y1 - 2.0*tt.log(1+ tt.exp(y2)) # simplified version

# 		y = pm.DensityDist('y', logp, shape=2, testval = [0,1])
# 		m = pm.Deterministic('m', pm.invlogit(y[0]))
# 		K = pm.Deterministic('K', pm.math.exp(y[1]))
# 		y_1 = pm.Deterministic('y_1', y[0])
# 		y_2 = pm.Deterministic('y_2', y[1])
		
# 		X0 = pm.BetaBinomial('X', alpha=K*m, beta=K*(1-m), n=n_data, observed=x_data)
		
# 		start = pm.find_MAP()
# 		inference = pm.NFVI('planar*4', jitter=1, start=start)
# 		approx = pm.fit(n=30000, method=inference)
# 		trace = approx.sample(draws=20000)

# 		m1, m2 = trace['y_1'], trace['y_2']

# 		m=sigmoid(m1)
# 		K=np.exp(m2)
# 		print('planar: m_mean:', np.mean(m))
# 		planar_m_means.append(np.mean(m))
# 		print('planar: m_mean:', np.std(m))
# 		print('planar: K_mean:', np.mean(K))
# 		planar_k_means.append(np.mean(K))
# 		print('planar: K_mean:', np.std(K))
# 		print('planar: cov:', np.cov(m,K))
# 		planar_covs.append(np.cov(m,K))


embed()
import sys
sys.path.append('/home/hadis/src/nips_pgm/code')

from distributions import density_1
from flows import ResnetFlow, PlanarFlow
from losses import elbo_loss, vanilla_vae_loss, elbo_loss_resnet, cross_entropy_loss, density_regeneration_loss
from mag.experiment import Experiment
from utils import plot_density, scatter_points, detJacHeatmap, deformationGrid, displacementField

import numpy as np
import tensorflow as tf
from IPython import embed

def logBetaBinomial(x,n,alpha,beta):
	def logComb(n,k):
		num = tf.lgamma(n+1)
		denum = tf.lgamma(k+1) + tf.lgamma(n-k+1)
		return tf.cast(num-denum, tf.float32)
	
	def betaln(x, y):
		return tf.cast(tf.lgamma(x) + tf.lgamma(y) - tf.lgamma(x + y), tf.float32)
		
#     val = bound(logComb(n,x) + betaln(x + alpha, n - x + beta) - betaln(alpha, beta),
#                     alpha > 0, beta > 0)
	val = logComb(n,x) + betaln(x + alpha, n - x + beta) - betaln(alpha, beta)
	return tf.cast(val,tf.float32)
   

def logModel(X,N,z):
	"""
	X : array of x data
	N : array of N data
	z : it is vector of y_1 and y_2
	"""
	y1, y2 = tf.split(z, [1,1], axis=1) 
	m = tf.sigmoid(y1)
	K = tf.exp(y2)
	logPrior = tf.cast(2.0*tf.log( 1 + tf.exp(-y1) ) +y1 - 2.0*tf.log(1+ tf.exp(y2)), tf.float32) 
	logp = 0
	for x,n in zip(X,N):
		logp = logp + logBetaBinomial(x, n , K*m, K*(1-m))
	
	logp = tf.reshape(logp,(-1,1)) + logPrior
	return logp


# experiment = Experiment({ "batch_size": 40,
# 						"iterations": 10000,
# 						"learning_rate": 0.01,
# 						"flow_length": 4,
# 						"flow": "resnet",
# 						"exp_name": "temp"})

# config = experiment.config
# experiment.register_directory("distributions")
# experiment.register_directory("postTrainingAnalysis")
# experiment.register_directory("samples")
# experiment.register_directory("tensorboard")
batch_size = 40
iterations = 10000
lr = 0.01 

import pandas as pd

data = pd.read_csv('cancermortality.csv')
x_data = np.array(data.y, dtype=np.float)
n_data = np.array(data.n, dtype=np.float)

# Other parameters
z_dim = 2
log_interval = 300

density_2 = lambda z: tf.exp(logModel(x_data,n_data,z))
# plot_density(density_2, directory=experiment.distributions, X_LIMS=(-8,-5),Y_LIMS=(4,8))

for _ in range(10):
	resnet_flow_graph = tf.Graph()

	with resnet_flow_graph.as_default():
		# ResnetFlow paramteres
		num_flows= 4
		num_cells_per_flow = 8
		dt = tf.constant(1.0/(num_cells_per_flow*num_flows), name='dt')
		approximation = 2 # 1 , 2 , 3 or 4 - None for exact calculation of the determinant of jacobian

		Z = tf.placeholder(tf.float32, shape =[None, 2], name='random_sample')

		with tf.variable_scope('Flow'):
			flow = ResnetFlow(num_flows, num_cells_per_flow, z_dim, dt, approximation=approximation)
			z_k, sum_log_detj, _ = flow.flow(Z)

		loss_op, density = density_regeneration_loss(density_2, z_k, sum_log_detj)
		tf.summary.scalar("loss", loss_op)

		global_step = tf.Variable(0, trainable=False, name='global_step')

		optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
		gvs = optimizer.compute_gradients(loss_op)
		# capped_gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), var) for grad, var in gvs]
		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		train_op = optimizer.apply_gradients(capped_gvs)

		# train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_op, global_step=global_step)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		# writer = tf.summary.FileWriter(experiment.tensorboard)
		summaries = tf.summary.merge_all()

		for iteration in range(1, iterations + 1):
			z_samples = np.random.normal(0.0, 1.0, [batch_size,2])
			_, loss, _,sldg, summ, den, grads = sess.run([train_op, loss_op, z_k, sum_log_detj, summaries, density, capped_gvs], feed_dict={Z: z_samples})
			step = tf.train.global_step(sess, tf.train.get_global_step())
			# print('sum_log_detj',str(sldg))
			if iteration % log_interval == 0:
				# writer.add_summary(summ, global_step=step)
				print("Loss on iteration {}: {}".format(iteration , loss))
				# z_samples = np.random.normal(0.0, 1.0, [1000,z_dim])
				# output = sess.run(z_k, feed_dict={Z: z_samples})
				# scatter_points(
				#     output,
				#     directory=experiment.samples,
				#     iteration=iteration,
				#     flow_length=config.flow_length,
				#     X_LIMS=(-8,-5), 
				#     Y_LIMS=(4,8)
				#     )
		
		# Draw samples from the trained flow model and plot the resutls
		z_samples = np.random.normal(0.0, 1.0, [20000,z_dim])
		samples = sess.run(z_k, feed_dict={Z: z_samples})
		m1, m2 = samples[:,0], samples[:,1]

		m=sigmoid(m1)
		K=np.exp(m2)
		print('our: m_mean:', np.mean(m))
		ours_m_means.append(np.mean(m))
		print('our: m_mean:', np.std(m))
		print('our: K_mean:', np.mean(K))
		ours_k_means.append(np.mean(m))
		print('our: K_mean:', np.std(K))
		print('our: cov:', np.cov(m,K))
		ours_covs.append(np.cov(m,K))

embed()