import numpy as np
import tensorflow as tf
from tf_utils import actnorm

from IPython import embed

class ResnetFlow(object):
	def __init__(self, n_flows, num_cells_per_flow, z_dim, dt, add_actnorm=False, approximation=1, regularize=0):
		self.num_flows = n_flows
		self.num_cells_per_flow = num_cells_per_flow
		self.z_dim = z_dim
		self.dt = dt
		self.approximation = approximation
		self.add_actnorm = add_actnorm
		self.regularize = regularize
	def resnet_block(self, z_in):
		w1 = tf.get_variable(name='weight1', shape=[self.z_dim, self.z_dim])
		b1 = tf.get_variable(name='bias1', shape=[self.z_dim])
		w2 = tf.get_variable(name='weight2', shape=[self.z_dim, self.z_dim], initializer=tf.zeros_initializer())
		b2 = tf.get_variable(name='bias2', shape=[self.z_dim], initializer=tf.zeros_initializer())

		h = tf.nn.tanh(tf.matmul(z_in, w1) + b1)
		h = tf.matmul(h, w2) + b2
		
		grad=[]

		for i in range(self.z_dim):
			grad+=(tf.gradients(ys=h[:, i], xs=z_in))
		return h, grad

	def flow(self, z0):
		if self.add_actnorm:
			z_in, logdetjac_actnorm = actnorm('actnorm', z0, logdet=0.0)
			sum_logdet_jacobian = logdetjac_actnorm
		else:
			z_in = z0
			sum_logdet_jacobian = 0.0

		velocity_norm_squared = tf.zeros_like(z0[:,0])
		for flow_index in np.arange(1, self.num_flows+1):
			with tf.variable_scope('flow'+str(flow_index), reuse=tf.AUTO_REUSE):
				for _ in range(self.num_cells_per_flow):
					res, jacobian = self.resnet_block(z_in)
					z_k = z_in + res*self.dt
					z_in = z_k
					jacobian = tf.stack(jacobian,axis=2)
					velocity_norm_squared += tf.reduce_sum(tf.square(res), 1)
					if self.approximation is None:
						logdet_jacobian = tf.log(tf.matrix_determinant(tf.eye(2) + self.dt*jacobian))

					if self.approximation == 1:
						logdet_jacobian = self.dt*tf.trace(jacobian)

					if self.approximation == 2:
						logdet_jacobian = self.dt*tf.trace(jacobian)

						JJ = jacobian * tf.transpose(jacobian, (0, 2, 1))
						logdet_jacobian -= 0.5* self.dt**2*tf.trace(JJ)

					if self.approximation == 3:
						m = 1
						r = tf.random_normal(shape=(self.z_dim,1))
						
						s = tf.map_fn(lambda j: tf.matmul(tf.transpose(r,(1,0)),tf.matmul(j,r)),jacobian)
						
						for ii in range(m-1):
							r = tf.random_normal(shape=(self.z_dim,1))
							s += tf.map_fn(lambda j: tf.matmul(tf.transpose(r,(1,0)),tf.matmul(j,r)),jacobian)
						s = s/m
						logdet_jacobian = self.dt*s[:,0,0]

					if self.approximation == 4:
						r0 = tf.random_normal(shape=(self.z_dim,1))				
						logdet_jacobian = tf.map_fn(lambda j: tf.matmul(tf.transpose(r0,(1,0)), self.dt*tf.matmul(tf.transpose(j),r0) - self.dt**2*tf.matmul(tf.matmul(tf.transpose(j),j), r0)),jacobian)
						logdet_jacobian[:,0,0]
				
					sum_logdet_jacobian += logdet_jacobian
		
		return  z_k, sum_logdet_jacobian, self.regularize * velocity_norm_squared
	
	def inverse(self, z_k):	
		for flow_index in np.arange(self.num_flows, 0, -1):
			with tf.variable_scope('flow'+str(flow_index), reuse=tf.AUTO_REUSE):
				for _ in range(self.num_cells_per_flow):
					res, jacobian = self.resnet_block(z_k)
					z_k = z_k - res*self.dt

		return z_k
		


class PlanarFlow():
	def __init__(self, z_dim, num_flows):
		self.z_dim = z_dim
		self.num_flows = num_flows
	def flow(self, z):
		sum_logdet_jacobian = 0.0*z[:,0]
		for k in range(self.num_flows):
			with tf.variable_scope('planar'+str(k+1)):
				w = tf.get_variable(name='weight', shape=[self.z_dim,1])
				b = tf.get_variable(name='bias', shape=[1])
				u = tf.get_variable(name='scale', shape=[self.z_dim,1])
				h = tf.nn.tanh(tf.matmul(z, w) + b)
				
				tf.summary.histogram('weight', w)				
				tf.summary.histogram('bias', b)				
				tf.summary.histogram('scale', u)				
				
				wTu = tf.matmul(tf.transpose(w),u)
				u_hat = u + (-1 + tf.keras.activations.softplus(wTu) - wTu)*w/tf.reduce_sum(w**2)
				z = z + h*tf.transpose(u_hat)

				# logdet jacobian calculation 
				psi = (1 - h ** 2) * tf.transpose(w)
				det_jac = 1 + tf.matmul(psi, u_hat)
				logdet_jac = tf.log(tf.abs(det_jac))
				sum_logdet_jacobian += logdet_jac[:,0]
		return z, sum_logdet_jacobian

	def inverse(self, z_k):	
		"""
		Needs implementation
		"""
		return z_k
