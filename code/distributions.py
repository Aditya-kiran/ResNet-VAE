import numpy as np
import tensorflow as tf 

def density_1(z):
	z1, z2 = tf.split(z, [1,1], axis=1) 
	norm = tf.sqrt(z1 ** 2 + z2 ** 2)
	exp1 = tf.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
	exp2 = tf.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
	u = 0.5 * ((norm - 4) / 0.4) ** 2 - tf.log(exp1 + exp2)
	return tf.exp(-u)

def density_2(z):
	z1, z2 = tf.split(z, [1,1], axis=1) 
	norm = tf.sqrt(z1 ** 2 + z2 ** 2)
	exp1 = tf.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
	exp2 = tf.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
	u = 0.5 * ((norm - 2) / 0.4) ** 2 - tf.log(exp1 + exp2)
	return tf.exp(-u)
