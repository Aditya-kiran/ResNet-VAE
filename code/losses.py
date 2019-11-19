import numpy as np
import tensorflow as tf
from IPython import embed

def cross_entropy_loss(prediction, actual, offset= 1e-4):
	with tf.name_scope("cross_entropy"):
		_prediction = tf.clip_by_value(prediction, offset, 1 - offset)

		if len(actual.shape)==2:
			ce_loss= - tf.reduce_sum(actual * tf.log(_prediction)
								 + (1 - actual) * tf.log(1 - _prediction), [1])
		if len(actual.shape)==4:
			ce_loss= - tf.reduce_sum(actual * tf.log(_prediction)
								 + (1 - actual) * tf.log(1 - _prediction), [1,2,3])
		return ce_loss
				

def kl_divergence_gaussian(mu, var):
	with tf.name_scope("kl_divergence"):
		_var = tf.clip_by_value(var, 1e-4, 1e6)
		kl = - 0.5 * tf.reduce_sum(1 + tf.log(_var) - tf.square(mu) - _var, 1)    
		return kl 
	

def gaussian_log_pdf(z, mu, var):
	return tf.contrib.distributions.MultivariateNormalDiag(
				loc = mu, scale_diag = tf.maximum(tf.sqrt(var), 1e-4)).log_prob(z + 1e-4)
	

def elbo_loss_resnet(actual, prediction, recons_error_func = cross_entropy_loss, **kwargs):
	mu= kwargs['z_mu']
	_var = kwargs['z_var']
	
	z0 = kwargs['z0']
	zk = kwargs['zk']
	logdet_jacobian = kwargs['logdet_jacobian']

	log_q0_z0 = gaussian_log_pdf(z0, mu, _var)
	sum_logdet_jacobian = logdet_jacobian
	log_qk_zk = log_q0_z0 - sum_logdet_jacobian

	log_p_x_given_zk =  recons_error_func(prediction, actual)
	log_p_zk =  gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu))
	
	recons_loss =  log_p_x_given_zk
	kl_loss = log_qk_zk - log_p_zk
	_elbo_loss = tf.reduce_mean(kl_loss + recons_loss)
	return _elbo_loss    


def elbo_loss(actual, prediction, recons_error_func=cross_entropy_loss, **kwargs):
	mu= kwargs['z_mu']
	_var = kwargs['z_var']
	
	z0 = kwargs['z0']
	zk = kwargs['zk']
	logdet_jacobian = kwargs['logdet_jacobian']

	log_q0_z0 = gaussian_log_pdf(z0, mu, _var)
	sum_logdet_jacobian = logdet_jacobian
	log_qk_zk = log_q0_z0 - sum_logdet_jacobian

	log_p_x_given_zk =  recons_error_func(prediction, actual)
	log_p_zk =  gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu))
	
	recons_loss =  log_p_x_given_zk
	kl_loss = log_qk_zk - log_p_zk
	_elbo_loss = tf.reduce_mean(kl_loss + recons_loss)
	return _elbo_loss    
	

def vanilla_vae_loss(x_reconstructed, x_true, z_mean, z_var):
	with tf.variable_scope('vanilla_vae_loss'):	
		reconstruction_Loss = cross_entropy_loss(prediction=x_reconstructed, actual=x_true)
		kl_div_loss = kl_divergence_gaussian(z_mean, z_var)
		returned_loss = tf.reduce_mean(reconstruction_Loss + kl_div_loss)
	return returned_loss


def log_normal(x, mean, var, eps=1e-5):
	const = - 0.5 * tf.log(2*np.pi)
	var += eps
	return const - tf.log(var)/2 - (x - mean)**2 / (2*var)

def safe_log(z):
    return tf.log(z + 1e-30)

def density_regeneration_loss(density, z_k, sum_of_log_jacobians):
	return tf.reduce_mean(-tf.reshape(sum_of_log_jacobians,(-1,1)) - safe_log(density(z_k))), density(z_k)

