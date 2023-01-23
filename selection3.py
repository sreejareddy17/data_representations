from matplotlib.pyplot import axis
from scipy import linalg
import numpy as np
import sys

arguments = sys.argv
if len(sys.argv) != 4:
    print("This program expects 4 arguments. Insufficient no of arguments")
    sys.exit(0)
data = np.array(np.loadtxt(arguments[1], delimiter=','))
labels = np.array(np.loadtxt(arguments[2], delimiter='\n'))
#def Fisher(data, data_lab):
temp = data
no_of_features = 2
for i in range(data.shape[1]-no_of_features):
    w,r,rank_mat,vals = linalg.lstsq(temp,labels)
    min_index = np.argmin(np.square(w))
    temp = np.delete(temp,min_index,axis=1)
np.savetxt(arguments[3],temp,delimiter=",")