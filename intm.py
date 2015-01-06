#iGeoNice to iGeo matrix
import theano
import theano.tensor as T
theano.config.exception_verbosity = 'high'
import numpy

def getINTMMatrix(rng, igeon_pose):
	
	
	np_fix_centr_to_origin = numpy.array([[1,0,-0.5],[0,1,-0.5],[0,0,1]])
	fix_centr_to_origin = theano.shared(value=np_fix_centr_to_origin, name='np_fix_centr_to_origin', borrow=True)

	np_trans_matrix = numpy.array([[1,0,t0],[0,1,t1],[0,0,1]])
	trans_matrix = theano.shared(value=np_trans_matrix, name='np_trans_matrix', borrow=True)

	np_scaling_matrix = numpy.array([[s0,1,1],[0,s1,1],[0,0,1]])
	scaling_matrix = theano.shared(value=np_scaling_matrix, name='np_scaling_matrix', borrow=True)

	np_shearing_matrix = numpy.array([[1,z,0],[0,1,1],[0,0,1]])
	shearing_matrix = theano.shared(value=np_shearing_matrix, name='np_shearing_matrix', borrow=True)

	np_rot_matrix = numpy.array([[cos_theta, -sin_theta, 0],[sin_theta,cos_theta,1],[0,0,1]])
	rot_matrix = theano.shared(value=np_rot_matrix, name='np_rot_matrix', borrow=True)

	np_fix_corner_to_origin = numpy.array([[1,0,0.5],[0,1,0.5],[0,0,1]])
	fix_corner_to_origin = theano.shared(value=np_fix_corner_to_origin, name='np_fix_corner_to_origin', borrow=True)

	igeo_pose = T.dot(fix_centr_to_origin, T.dot(trans_matrix, T.dot(scaling_matrix, T.dot(shearing_matrix, T.dot(rot_matrix, T.dot(fix_corner_to_origin))))))

	return igeo_pose