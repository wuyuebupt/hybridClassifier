import os,sys

import numpy as np
from sklearn import neighbors, datasets
from numpy import linalg as LA
import scipy
import cPickle as cp 




if __name__ =='__main__':
	# build knn
	lines = open (sys.argv[1])
	fprefix = 'front_data/knn_feats/'
	bprefix = 'back_data/knn_feats/'
	x = []
	y = []
	for line in lines:
		arr = line.strip().split()
		imgpath = arr[0]
		ffeatpath = fprefix + imgpath + '.feat.bin'
		bfeatpath = bprefix + imgpath + '.feat.bin'
		ffeat_ = np.fromfile(ffeatpath, dtype=np.dtype('f4'))
		bfeat_ = np.fromfile(bfeatpath, dtype=np.dtype('f4'))
		# ffeat_ = np.fromfile(ffeatpath, dtype=np.dtype('f4')).reshape([512,1])
		# bfeat_ = np.fromfile(bfeatpath, dtype=np.dtype('f4')).reshape([512,1])
		# print ffeat_.shape
		feat_ = np.concatenate((ffeat_, bfeat_))
		# print feat_.shape
		feat_ = feat_ / LA.norm(feat_)

		label = int(arr[1])
		x.append(feat_)
		y.append(label)


	clf = neighbors.NearestNeighbors(1)
	clf.fit(x)

	fpredprefix = 'front_data/validation/'
	bpredprefix = 'back_data/validation/'
	lines = open(sys.argv[2])
	thres = float(sys.argv[3])
	for line in lines:
		arr = line.strip().split()
		imgpath = arr[-1]
		ffeatpath = fpredprefix + arr[-1]+'.feat.bin'
		bfeatpath = bpredprefix + arr[-1]+'.feat.bin'
		ffeat_ = np.fromfile(ffeatpath, dtype=np.dtype('f4'))
		bfeat_ = np.fromfile(bfeatpath, dtype=np.dtype('f4'))
		feat_ = np.concatenate((ffeat_, bfeat_))
		feat_ = feat_ / LA.norm(feat_)
		feat_ = feat_.reshape(1,-1)
		[dis,ind] = clf.kneighbors(feat_,1)
		ind_ = ind[0][0]
		knn = x[ind_]
		cos = 1-np.dot(feat_ - knn, (feat_ - knn).T)/2
		pred_ = y[ind_]
		if cos[0][0] >= thres or float(arr[2]) < cos[0][0]:
		# print pred_, cos[0][0], arr[1], imgpath
			print pred_, arr[1], cos[0][0], imgpath
		else:
			# print arr[0], arr[1], arr[2], imgpath
			print arr[0], arr[1], cos[0][0], imgpath
	
	# f = open('sub_train_test.pkl', 'wb')
	# cp.dump((X,y, testX, testy, videoids, frameids, vfindex), f)


	
