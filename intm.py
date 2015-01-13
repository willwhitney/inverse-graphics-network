#iGeoNice to iGeo matrix
import theano
import theano.tensor as T
# theano.config.exception_verbosity = 'high'
import pdb,math
import numpy as np

# def getINTMMatrix(x, bsize, rng, igeon_pose):
def getINTMMatrix(xx,bsize, rng, igeon_pose):

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

  #correct but very very slow
  igeo_pose = T.zeros([bsize, 3, 3])
  transformations = [fix_center_to_origin, trans_matrix, scaling_matrix, shearing_matrix, rot_matrix, fix_corner_to_origin]
  
  for ii in range(bsize):
    igeo_pose = T.set_subtensor(igeo_pose[ii,:,:], transformations[0][ii,:,:])
    for jj in range(1,len(transformations)):
      igeo_pose = T.set_subtensor(igeo_pose[ii,:,:], T.dot(transformations[jj][ii,:,:], igeo_pose[ii,:,:]))
  igeo_pose_gt = T.batched_dot(fix_corner_to_origin, T.batched_dot(rot_matrix, T.batched_dot(shearing_matrix, T.batched_dot(scaling_matrix, T.batched_dot(trans_matrix, fix_center_to_origin)))))


  # n1 = fix_center_to_origin.eval()[0];
  # n2 = trans_matrix.eval()[0];
  # n3 = scaling_matrix.eval()[0];
  # n4 = shearing_matrix.eval()[0];
  # n5 = rot_matrix.eval()[0];
  # n6 = fix_corner_to_origin.eval()[0];
  # npres = np.dot(n1,np.dot(n2,np.dot(n3,np.dot(n4, np.dot(n5,n6)))))

  pdb.set_trace()
  return (igeo_pose, intensity)










