import numpy as np
import pickle

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        objs = []
        while 1:
            try:
                o = pickle.load(f)
            except EOFError:
                break
            objs.append(o)
    return objs

l = [1,0,0,4,2,1,0,2,5,4,2,1,4,0,1,5,4,0,1,4,2,0,2,1,0,0,4,4,5,0,2,5,1,0,5,0,5,5,3,5,5,0,4,1,2,1,0,2,5,0,0,2,1,0,0,5,1,2,1,5,2,2,4,5,5,4,1,3]

k= []
k.append([i for i, x in enumerate(l) if x == 0])
k.append([i for i, x in enumerate(l) if x == 1])
k.append([i for i, x in enumerate(l) if x == 2])
k.append([i for i, x in enumerate(l) if x == 3])
k.append([i for i, x in enumerate(l) if x == 4])
k.append([i for i, x in enumerate(l) if x == 5])


u = load_pickle('U_processed.p')
lfs = u[2]
b = np.zeros((lfs.shape[0], 6))
for i in range(lfs.shape[0]):
	for j in range(6):
		if(len(lfs[i][k[j]])[0]) > 0):
			b[i][j] = 1