import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot


def calculate_mixtureMean(X):
    res = np.mean(X, axis=0)
    return res


def calculate_mixtureClassScatter(X, Y):
    ans = np.dot((X - Y).T, (X - Y))
    return ans


def calculate_withinClassScatter(group1, group2, group3):
    mean_group1 = np.mean(group1, axis=0)
    mean_group2 = np.mean(group2, axis=0)
    mean_group3 = np.mean(group3, axis=0)
    diff1 = group1 - mean_group1
    diff2 = group2 - mean_group2
    diff3 = group3 - mean_group3
    scatter1 = np.dot(diff1.T, diff1)
    scatter2 = np.dot(diff2.T, diff2)
    scatter3 = np.dot(diff3.T, diff3)
    return scatter1 + scatter2 + scatter3


arguments = sys.argv
if len(arguments) != 4:
    print("This program expects 4 arguments. Insufficient no of arguments")
    sys.exit(0)

data = np.loadtxt(arguments[1], delimiter=",")
labels = np.loadtxt(arguments[2], delimiter=",")
df_data = pd.DataFrame(data)
df_labels = pd.DataFrame(labels)
group1 = data[np.where(labels == 1)]
group2 = data[np.where(labels == 2)]
group3 = data[np.where(labels == 3)]
mixture_mean = calculate_mixtureMean(data)
mixtureClass_scatter = calculate_mixtureClassScatter(data, mixture_mean)
withinClass_scatter = calculate_withinClassScatter(group1, group2, group3)
mean_group1 = np.mean(group1, axis=0).reshape(4, 1)
mean_group2 = np.mean(group2, axis=0).reshape(4, 1)
mean_group3 = np.mean(group3, axis=0).reshape(4, 1)
mixture_mean = mixture_mean.reshape(4, 1)
a = mean_group1 - mixture_mean
b = mean_group2 - mixture_mean
c = mean_group3 - mixture_mean
betweenClassScatter = len(group1) * (np.dot(a, a.T)) + len(group2) * (np.dot(b, b.T)) + len(group3) * (np.dot(c, c.T))
# print(betweenClassScatter)
eigen_values, eigen_vectors = np.linalg.eig(mixtureClass_scatter)
list_evalues = list(eigen_values)
list_evectors = list(eigen_vectors)
lis_maximum = []
for i in range(2):
    x = max(list_evalues)
    lis_maximum.append(list_evalues.index(x))
    list_evalues.pop(list_evalues.index(x))
res = []
res.append(eigen_vectors[:,lis_maximum[0]])
res.append(eigen_vectors[:,lis_maximum[1]])
output_data = np.dot(res,data.T)
np.savetxt(arguments[3],output_data.T,delimiter=",")
plot.scatter(output_data.T[:,0],output_data.T[:,1])
plot.show()