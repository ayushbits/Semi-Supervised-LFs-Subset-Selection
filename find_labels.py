import pickle
import numpy as np
objs = []

with open('Data/CENSUS/d_processed.p', 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

labs = objs[1].T
lfs = labs.shape[0]
num_labs = 2
final_labs = []
for i  in range(lfs):
	for j in range(num_labs):
		x = np.where(labs[i]==j)
		if (len(x[0]) > 0):
			final_labs.append(j)


print(final_labs)