from pymongo import MongoClient
from keras.models import load_model
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pymongo

expkey='Crab64080opt3'
c = MongoClient('mongodb://exet4487:admin123@192.168.0.200:27017/jobs')
best_model = c['jobs']['jobs'].find_one({'exp_key': expkey,'result.status':'ok'}, sort=[('result.loss',pymongo.ASCENDING)])


#raise KeyboardInterrupt
j=0
losses=[]
modelnos=[]

for i in c['jobs']['jobs'].find({'exp_key':expkey,'result.status':'ok'}):
    if j>344:
        break
    
    print(i)
    print(dir(i['result']))
    print(i['result'].keys())
    losses.append(i['result']['loss'])
    modelnos.append(i['result']['modelno'])
    j=j+1
    print(j)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]/n

def running_average(a):
    avgn=np.zeros(len(a))
    for i in np.arange(len(a))[1:]:
        j=int(i)
        avgn[j]=np.mean(a[0:j])
    return avgn

losses=-1.0*np.asarray(losses)
avg7=moving_average(losses,n=7)
avg9=moving_average(losses,n=9)
avgn=running_average(losses)

plt.plot(np.arange(j),losses,label='Binary Accuracy',marker='+',linestyle='None')
plt.plot(np.arange(len(avgn))[1:],avgn[1:],label='Running Average')
#plt.plot(np.arange(len(avg7)),avg7,label='Moving average, n=7')
plt.plot(np.arange(len(avg9)),avg9,label='Moving Average, n=9')

plt.xlabel('Iteration')
plt.ylabel('Binary Accuracy')
plt.title('Diffuse Training Optimisation')
plt.legend()
plt.savefig('/users/exet4487/Figures/convplot_'+expkey+'.png')
#print(losses,modelnos)

percents=[50,55,60,65,70,75,80,85,90,95,100]

print('Percentiles:')
for p in percents:
    i_near=abs(losses-np.percentile(losses,p,interpolation='nearest')).argmin()
    print(p,i_near,losses[i_near],modelnos[i_near])

print(best_model)
