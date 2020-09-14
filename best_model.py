from pymongo import MongoClient
from keras.models import load_model
import tempfile
import matplotlib.pyplot as plt
import numpy as np

expkey='longtest7'
c = MongoClient('mongodb://exet4487:admin123@192.168.0.200:27017/jobs')
best_model = c['jobs']['jobs'].find_one({'exp_key': expkey}, sort=[('result.loss', -1)])

print(best_model)
#raise KeyboardInterrupt
j=0
losses=[]

for i in c['jobs']['jobs'].find({'exp_key':expkey,'result.status':'ok'}):
    print(i['result'])
    losses.append(i['result']['loss'])
    j=j+1
    print(j)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]/n

losses=-1.0*np.asarray(losses)
avg7=moving_average(losses,n=7)
avg9=moving_average(losses,n=9)
plt.plot(np.arange(j),losses,label='Binary Accuracy')

plt.plot(np.arange(len(avg7)),avg7,label='Moving average, n=7')
plt.plot(np.arange(len(avg9)),avg9,label='Moving average, n=9')

plt.xlabel('Iteration')
plt.ylabel('Binary Accuracy')
plt.legend()
plt.savefig('/users/exet4487/Figures/convplot_'+expkey+'.png')
