#iGeoNice to iGeo matrix
import theano
import theano.tensor as T
# theano.config.exception_verbosity = 'high'
import pdb,math
import numpy as np
#  0  1  2 3 4 5 6
# [1,-3,-3,1,1,0,0*math.pi/2]

def build_single_iGeoPose(single_igeon):
	# return single_igeon
	# pdb.set_trace()
	res = T.zeros([3,3])
	res = T.set_subtensor(res[0,0],
		single_igeon[3] * T.cos(single_igeon[6]))
	res = T.set_subtensor(res[0,1],
		single_igeon[4] * (single_igeon[5] * T.cos(single_igeon[6]) - T.sin(single_igeon[6])))
	res = T.set_subtensor(res[0,2],
		single_igeon[3] * single_igeon[1] * T.cos(single_igeon[6]) + single_igeon[4] * single_igeon[2] * (single_igeon[5] * T.cos(single_igeon[6]) - T.sin(single_igeon[6])))
	res = T.set_subtensor(res[1,0],
		single_igeon[3] * T.sin(single_igeon[6]))
	res = T.set_subtensor(res[1,1],
		single_igeon[4] * (single_igeon[5] * T.sin(single_igeon[6]) + T.cos(single_igeon[6])))
	res = T.set_subtensor(res[1,2],
		single_igeon[3] * single_igeon[1] * T.sin(single_igeon[6]) + single_igeon[4] * single_igeon[2] * (single_igeon[5] * T.sin(single_igeon[6]) + T.cos(single_igeon[6])))
	res = T.set_subtensor(res[2,2], T.constant(1.))
	return res

# def getINTMMatrix(x, bsize, rng, igeon_pose):
def getINTMMatrix(bsize, rng, igeon_pose):

	#xx={x:np.float32(np.random.rand(2,28*28))}

	intensity = igeon_pose[:,0]
	t0=igeon_pose[:,1]; t1=igeon_pose[:,2]
	s0=igeon_pose[:,3]; s1=igeon_pose[:,4]
	z=igeon_pose[:,5];
	# cos_theta=T.cos(igeon_pose[:,6]); sin_theta=T.sin(igeon_pose[:,6])
	theta = igeon_pose[:, 6]

	image_size = 0#15.0
	offset = image_size / 2# - 2.0/2

	fix_center_to_origin = T.zeros([bsize, 3, 3])
	fix_center_to_origin = T.inc_subtensor(fix_center_to_origin[:,0,0], T.constant(1.0))
	fix_center_to_origin = T.inc_subtensor(fix_center_to_origin[:,0,2], T.constant(-offset))
	fix_center_to_origin = T.inc_subtensor(fix_center_to_origin[:,1,1], T.constant(1.0))
	fix_center_to_origin = T.inc_subtensor(fix_center_to_origin[:,1,2], T.constant(-offset))
	fix_center_to_origin = T.inc_subtensor(fix_center_to_origin[:,2,2], T.constant(1.0))

	trans_matrix = T.zeros([bsize, 3 , 3])
	trans_matrix = T.inc_subtensor(trans_matrix[:,0,0], T.constant(1.0))
	trans_matrix = T.inc_subtensor(trans_matrix[:,1,1], T.constant(1.0))
	trans_matrix = T.inc_subtensor(trans_matrix[:,2,2], T.constant(1.0))
	trans_matrix = T.inc_subtensor(trans_matrix[:,0,2], t0)
	trans_matrix = T.inc_subtensor(trans_matrix[:,1,2], t1)

	scaling_matrix = T.zeros([bsize,3,3])
	scaling_matrix = T.inc_subtensor(scaling_matrix[:,0,0], s0)
	scaling_matrix = T.inc_subtensor(scaling_matrix[:,1,1], s1)
	# scaling_matrix = T.inc_subtensor(scaling_matrix[:,1,2], T.constant(1.0))
	scaling_matrix = T.inc_subtensor(scaling_matrix[:,2,2], T.constant(1.0))

	shearing_matrix = T.zeros([bsize,3 ,3])
	shearing_matrix = T.inc_subtensor(shearing_matrix[:,0,1], z)
	shearing_matrix = T.inc_subtensor(shearing_matrix[:,0,0], T.constant(1.0))
	shearing_matrix = T.inc_subtensor(shearing_matrix[:,1,1], T.constant(1.0))
	# shearing_matrix = T.inc_subtensor(shearing_matrix[:,1,2], T.constant(1.0))
	shearing_matrix = T.inc_subtensor(shearing_matrix[:,2,2], T.constant(1.0))

	rot_matrix = T.zeros([bsize,3,3])
	rot_matrix = T.inc_subtensor(rot_matrix[:,0,0], T.cos(theta))
	rot_matrix = T.inc_subtensor(rot_matrix[:,0,1], -T.sin(theta))
	rot_matrix = T.inc_subtensor(rot_matrix[:,1,0], T.sin(theta))
	rot_matrix = T.inc_subtensor(rot_matrix[:,1,1], T.cos(theta))
	# rot_matrix = T.inc_subtensor(rot_matrix[:,1,2], T.constant(1.0))
	rot_matrix = T.inc_subtensor(rot_matrix[:,2,2], T.constant(1.0))

	fix_corner_to_origin = T.zeros([bsize, 3, 3])
	fix_corner_to_origin = T.inc_subtensor(fix_corner_to_origin[:,0,0], T.constant(1.0))
	fix_corner_to_origin = T.inc_subtensor(fix_corner_to_origin[:,0,2], T.constant(offset))
	fix_corner_to_origin = T.inc_subtensor(fix_corner_to_origin[:,1,1], T.constant(1.0))
	fix_corner_to_origin = T.inc_subtensor(fix_corner_to_origin[:,1,2], T.constant(offset))
	fix_corner_to_origin = T.inc_subtensor(fix_corner_to_origin[:,2,2], T.constant(1.0))

	#igeo_pose = T.batched_dot(fix_center_to_origin, T.batched_dot(trans_matrix, T.batched_dot(scaling_matrix, T.batched_dot(shearing_matrix, T.batched_dot(rot_matrix, fix_corner_to_origin)))))
	igeo_pose = T.batched_dot(fix_corner_to_origin, T.batched_dot(rot_matrix, T.batched_dot(shearing_matrix, T.batched_dot(scaling_matrix, T.batched_dot(trans_matrix, fix_center_to_origin)))))

	# pdb.set_trace()

	res, _ = theano.map(lambda i: build_single_iGeoPose(igeon_pose[i, :]),
						 sequences=[T.arange(bsize)])
	# res, _ = theano.map(lambda i: igeon_pose[i, :],
	# 					 sequences=[T.arange(bsize)])
	# n1 = fix_center_to_origin.eval()[0];
	# n2 = trans_matrix.eval()[0];
	# n3 = scaling_matrix.eval()[0];
	# n4 = shearing_matrix.eval()[0];
	# n5 = rot_matrix.eval()[0];
	# n6 = fix_corner_to_origin.eval()[0];
	# npres = np.dot(n1,np.dot(n2,np.dot(n3,np.dot(n4, np.dot(n5,n6)))))
	pdb.set_trace()
	return (igeo_pose, intensity)

  # for ii in range(bsize):
  #   igeo_pose = T.set_subtensor(igeo_pose[ii,:,:], transformations[0][ii,:,:])
  #   for jj in range(1,len(transformations)):
  #     igeo_pose = T.set_subtensor(igeo_pose[ii,:,:], T.dot(transformations[jj][ii,:,:], igeo_pose[ii,:,:]))
  # igeo_pose_gt = T.batched_dot(fix_corner_to_origin, T.batched_dot(rot_matrix, T.batched_dot(shearing_matrix, T.batched_dot(scaling_matrix, T.batched_dot(trans_matrix, fix_center_to_origin)))))











