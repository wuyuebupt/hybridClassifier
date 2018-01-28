import os,sys

if __name__ == '__main__':
	lines = open(sys.argv[1])
	mapping  = {}
	for line in lines:
		arr = line.strip().split()
		mapping[arr[1]] = arr[0]

	lines = open(sys.argv[2])
	for line in lines:
		arr = line.strip().split()
		lineid = arr[-1].split('/')[1].split('.')[0]
		outline = lineid + '\t' + mapping[arr[0]] + '\t' + arr[2]
		print outline
		
