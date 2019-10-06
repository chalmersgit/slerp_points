'''
Andrew Chalmers, 2019

Resource:
https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def slerp_points(a,b,nPointsPerDegree=1,showPlot=False):
	"""
	Computes N spherically interpolated points between two 3D points (a and b)
	where N scales based on the angle between a and b using nPointsPerDegree.
	Parameters
	----------
	a : ndarray (1,3)
		The point to slerp from.
	b : ndarray (1,3)
		The point to slerp to.
	nPointsPerDegree : float, optional
		The number of points to generate per degree between a and b.
	showPlot : bool, optional
	    Whether to display the points in a 3D plot.
	Returns
	-------
	points : ndarray (N,3)
		Returns (N,3) shape, where each row is a slerped point between a and b.
	Example
	--------
	points = slerp_points(np.asarray([0,1,0]), np.asarray([1,0,0]),0.5,True)
	"""

	# Helper function
	def normalize(x):
		if np.dot(x,x)==0: # avoid origin
			return x
		return x/np.linalg.norm(x)

	# Avoid origin
	if np.dot(a,a)==0:
		return np.asarray([b,])
	if np.dot(b,b)==0:
		return np.asarray([a,])

	# Normalize input points
	a = normalize(a)
	b = normalize(b)

	# Make perpendicular vector
	if np.abs(np.dot(a,b))>=1.0: 
		# account for parralel vectors: be careful of the direction of the arc
		if a[1]!=1:
			k = normalize(np.cross(a,[0,1,0]))
		else:
			k = normalize(np.cross(a,[1,0,0]))
	else:
		k = normalize(np.cross(a,b))

	# Angle between a and b
	theta = np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))
	
	# Generate angles, precompute sin and cos
	N = int((theta*57.2958)*nPointsPerDegree)
	if N<=0:
		N = 1
	angles = np.linspace(0,theta,N)
	angles_cos = np.cos(angles)[:,None]
	angles_sin = np.sin(angles)[:,None]

	# Tile data for vectorisation
	a_tiled = np.tile(a, (N,1))
	k_tiled = np.tile(k, (N,1))
	kXa_tiled = np.tile(np.cross(k,a), (N,1))
	kDa_tiled = np.tile(np.dot(k,a), (N,1))

	# Compute N points between a and b
	points = a_tiled*angles_cos+kXa_tiled*angles_sin+k_tiled*kDa_tiled*(1.0-angles_cos)

	if showPlot:
		fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal', 'facecolor':'gray'})
		ax.scatter(0,0,0, s=10, c='k')
		ax.scatter(a[0], a[1], a[2], s=10, c='r')
		ax.scatter(b[0], b[1], b[2], s=10, c='g')
		ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c='b')
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		plt.show()

	return points

