import numpy as np
import sys
import pdb
# pdb.set_trace()
arguments = sys.argv
if len(arguments)!=4:
    print("This program expects 4 arguments. Insufficient no of arguments")
    sys.exit(0)
data = np.array(np.loadtxt(arguments[1], delimiter=","))
#row, cols = data.shape
labels = np.array(np.loadtxt(arguments[2], delimiter="\n"))
Muj = np.mean(data,axis=0)
X = data-Muj
Sj = np.sum(np.square(X))
Muy = np.mean(labels,axis=0)
Y = labels-Muy
Sy = np.sum(np.square(Y))
Cjy = np.dot(X.T,Y)
#print(Cjy)
val = np.sqrt(Sj*Sy)
pearson_coefficient = np.abs(Cjy/val)
pearson_coefficient = list(pearson_coefficient)
#print(pearson_coefficient)
features = sorted(range(len(pearson_coefficient)), key=lambda sub: pearson_coefficient[sub])[-2:]
with open(sys.argv[3],'w') as f:
    f.write(str(features))

