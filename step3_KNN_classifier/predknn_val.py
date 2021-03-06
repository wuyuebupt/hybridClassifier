import os,sys

import numpy as np
from sklearn import neighbors, datasets
from numpy import linalg as LA
import scipy
import cPickle as cp 




if __name__ =='__main__':
	# build knn
	lines = open (sys.argv[1])
	newprefix = 'front_data/knn_feats/'
	x = []
	y = []
	for line in lines:
		arr = line.strip().split()
		imgpath = arr[0]
		featpath = newprefix + imgpath + '.feat.bin'
		feat_ = np.fromfile(featpath, dtype=np.dtype('f4'))
		feat_ = feat_ / LA.norm(feat_)

		label = int(arr[1])
		x.append(feat_)
		y.append(label)


	clf = neighbors.NearestNeighbors(1)
	clf.fit(x)

	predprefix = 'front_data/validation/'
	lines = open(sys.argv[2])
	for line in lines:
		arr = line.strip().split()
		imgpath = arr[0]
		featpath = predprefix + arr[0]+'.feat.bin'
		feat_ = np.fromfile(featpath, dtype=np.dtype('f4'))
		feat_ = feat_ / LA.norm(feat_)
		feat_ = feat_.reshape(1,-1)
		[dis,ind] = clf.kneighbors(feat_,1)
		ind_ = ind[0][0]
		knn = x[ind_]
		cos = 1-np.dot(feat_ - knn, (feat_ - knn).T)/2
		pred_ = y[ind_]
		# print pred_, cos[0][0], arr[1], imgpath
		print pred_, arr[1], cos[0][0], imgpath
	# f = open('sub_train_test.pkl', 'wb')
	# cp.dump((X,y, testX, testy, videoids, frameids, vfindex), f)


	
