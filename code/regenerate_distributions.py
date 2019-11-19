from distributions import density_1, density_2
from flows import ResnetFlow, PlanarFlow
from losses import elbo_loss, vanilla_vae_loss, elbo_loss_resnet, cross_entropy_loss, density_regeneration_loss
from mag.experiment import Experiment
from utils import plot_density, scatter_points, detJacHeatmap, deformationGrid, displacementField, \
				plot_density_from_samples, copy_files

import argparse
import numpy as np
import tensorflow as tf
from IPython import embed
import  matplotlib.pyplot as plt
import os

DENSITIES = [density_1, density_2]

tf_config = tf.ConfigProto()
tf_config.log_device_placement=False
tf_config.allow_soft_placement = True
tf_config.gpu_options.allow_growth=True
tf_config.gpu_options.per_process_gpu_memory_fraction = 1


parser = argparse.ArgumentParser()
parser.add_argument( '--flow', type=str, required=True, help='The length of the flow')
parser.add_argument( '--flow-length', type=int, default=1, help='The length of the flow')
parser.add_argument( '--num-cells', type=int, default=1, help='The number of cells per flow (only applicable with resnet)')
parser.add_argument( '--dir', type=str, default='./experiments', help='The directory to save results')
parser.add_argument( '--regularize', type=float, default=0.0, help='Regularization constant of the flow (only applicable for resnet)')
parser.add_argument( '--density', type=int, default=1, help='which density function to regenerate')
parser.add_argument( '--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument( '--iters', type=int, default=20000, help='Total number of iterations')
parser.add_argument( '--seed', type=int, default=1, help='Random seed')

args = parser.parse_args()

tf.set_random_seed(args.seed)

experiment = Experiment({ "batch_size": 40,
						"iterations": args.iters,
						"learning_rate": args.lr,
						"flow_length": args.flow_length,
						"flow": args.flow,
						"Regularization": args.regularize,
						"density": args.density,
						"seed": args.seed},
						experiments_dir=args.dir
						)

density = DENSITIES[args.density-1]

config = experiment.config
experiment.register_directory("distributions")
experiment.register_directory("postTrainingAnalysis")
experiment.register_directory("samples")
experiment.register_directory("reconstructed_distribution")
experiment.register_directory("invertedSamples")
experiment.register_directory("tensorboard")
experiment.register_directory("code_directory")

copy_files(experiment.code_directory)

# Other parameters
z_dim = 2
log_interval = 300

# ResnetFlow paramteres
if config.flow == 'resnet':
	num_flows= config.flow_length
	num_cells_per_flow = args.num_cells
	dt = tf.constant(1.0/(num_cells_per_flow*num_flows), name='dt')
	approximation = 2 # 1 , 2 , 3 or 4 - None for exact calculation of the determinant of jacobian

plot_density(density, directory=experiment.distributions)

Z = tf.placeholder(tf.float32, shape =[None, 2], name='random_sample')

with tf.variable_scope('Flow'):
	if config.flow == 'resnet':
		flow = ResnetFlow(num_flows, num_cells_per_flow, z_dim, dt, approximation=approximation, regularize=args.regularize)
		z_k, sum_log_detj, velocity_norm_squared = flow.flow(Z)
		z0_regenerated = flow.inverse(z_k)

	if config.flow == 'planar':
		flow = PlanarFlow(z_dim, num_flows=config.flow_length)
		z_k, sum_log_detj = flow.flow(Z)
		z0_regenerated = flow.inverse(z_k)
		velocity_norm_squared = tf.zeros_like(z_k[:,0])

loss_op,_ = density_regeneration_loss(density, z_k, sum_log_detj)
tf.summary.scalar("loss", loss_op)

global_step = tf.Variable(0, trainable=False, name='global_step')

train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_op + tf.reduce_mean(velocity_norm_squared), 
		global_step=global_step)

sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(experiment.tensorboard)
summaries = tf.summary.merge_all()

for iteration in range(config.iterations):
	z_samples = np.random.normal(0.0, 1.0, [config.batch_size,2])		
	_, loss, summ = sess.run([train_op, loss_op, summaries], feed_dict={Z: z_samples})
	step = tf.train.global_step(sess, tf.train.get_global_step())

	if iteration % log_interval == 0:
		writer.add_summary(summ, global_step=step)
		print("Loss on iteration {}: {}".format(iteration , loss))

		z_samples = np.random.normal(0.0, 1.0, [1000,z_dim])		
		output,inverted_output = sess.run([z_k, z0_regenerated], feed_dict={Z: z_samples})
		plot_density_from_samples(			
			samples=output,
			directory=experiment.reconstructed_distribution,
			iteration=iteration,
			flow_length=config.flow_length,
			)

		scatter_points(
			points=output,
			directory=experiment.samples,
			iteration=iteration,
			flow_length=config.flow_length)

		scatter_points(
			points=inverted_output,
			directory=experiment.invertedSamples,
			iteration=iteration,
			flow_length=config.flow_length,
			reference_points=z_samples)

z_samples = np.random.normal(0.0, 1.0, [10000,z_dim])		
output,inverted_output = sess.run([z_k, z0_regenerated], feed_dict={Z: z_samples})
plot_density_from_samples(			
	samples=output,
	directory=experiment.reconstructed_distribution,
	iteration=-9999,
	flow_length=config.flow_length,
	)

def headline_print(title):
	print('###############################')
	print(title)
	print('###############################')

# Plot heatmap of detjac and logdetjac
headline_print('Plotting heatmap of detjac and logdetjac ...')
flowFcn = lambda x: sess.run(z_k, feed_dict={Z: x[np.newaxis,:]})
domain_limits = (-6,6)
jacVals, ratio_negative = detJacHeatmap([(domain_limits[0],domain_limits[1],30),(domain_limits[0],domain_limits[1],30)], 
										flowFcn, 
										displaceFlag=False)

plt.figure()
plt.imshow(jacVals, 
	extent=(domain_limits[0], domain_limits[1], domain_limits[0], domain_limits[1]), 
	origin='lower',
	cmap='viridis')
plt.colorbar()
plt.title("Determinant of the Flow Jacobian \n Percentage of negative detjac is " + str(100*ratio_negative) + "%")
plt.savefig(os.path.join(experiment.postTrainingAnalysis,'detjac_heatmap.png'))
plt.close()

plt.figure()
plt.imshow(np.log(jacVals), 
	extent=(domain_limits[0], domain_limits[1], domain_limits[0], domain_limits[1]), 
	origin='lower',
	cmap='viridis')
plt.colorbar()
plt.title("Log Determinant of the Flow Jacobian \n Percentage of undefined logdetjac is " + str(100*ratio_negative) + "%")
plt.savefig(os.path.join(experiment.postTrainingAnalysis,'logdetjac_heatmap.png'))
plt.close()

# Deformation Grid
headline_print('Plotting deformation grids ...')
flowFcn = lambda x: sess.run(z_k, feed_dict={Z: x})
domainRange = [(-6,6),(-6,6)]
gridSize = [20,20]
numPoints = 100
deformationGrid(flowFcn, 
				domainRange, 
				gridSize, 
				numPoints, 
				experiment.postTrainingAnalysis)        

# DisplacementField
headline_print('Plotting displacement fields ...')
domainRange = [(-6,6,20),(-6,6,20)]
visualizeRange = [(-10,10),(-10,10)]
displacementField(flowFcn,
				domainRange,
				visualizeRange, 
				experiment.postTrainingAnalysis)

# DisplacementField after inversion
flowFcn2 = lambda x: sess.run(z0_regenerated, feed_dict={Z: x})
domainRange = [(-6,6,20),(-6,6,20)]
visualizeRange = [(-10,10),(-10,10)]
displacementField(flowFcn2,
				domainRange,
				visualizeRange, 
				experiment.postTrainingAnalysis,
				title='InversionDisplacementField')

