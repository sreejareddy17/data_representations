import numpy as np

Rows = int(input("Give the number of rows:"))
Columns = int(input("Give the number of columns:"))

print("Please write the elements of the matrix in a single line and separated by a space: ")

# User will give the entries in a single line
elements = list(map(float, input().split()))

# Printing the matrix given by the user
X = np.array(elements).reshape(Rows, Columns)

evals,evtrs = np.linalg.eig(X)
print(evals,evtrs)