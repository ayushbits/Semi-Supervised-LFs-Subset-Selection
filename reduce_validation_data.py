import pickle

objs=[]
with open('data/CENSUS/validation_processed.p', 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x = []

write_instance = 83
for i in range(len(objs)):
    x.append(objs[i][0:write_instance])

with open('data/CENSUS/reduce_validation.p','wb') as f:
	for val in x:
		pickle.dump(val,f)
