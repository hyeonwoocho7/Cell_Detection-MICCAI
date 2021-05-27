import numpy as np
import matplotlib.pyplot as plt
import os

path1 = ["B4_1_00001_002_000.tif", "B4_1_00016_003_001.tif", "B4_1_00082_002_007.tif", "B4_1_00086_000_007.tif","B4_1_00087_003_005.tif",
         "B4_1_00090_001_007.tif", "B4_1_00096_002_005.tif", "B4_1_00099_005_003.tif"]
path = "/home/hyeonwoo/research/Experiment1/Result/Discriminator/step1/sigmoid_distribution/sigmoid.txt"
f = open(path, 'r')
data = f.read()
datas = data.split("\n")
datas = datas[:-1]
paths = []
values = []
for i, data in enumerate(datas):
    data = data.split(', ')
    paths.append(os.path.basename(data[1]))
    values.append(data[0])
k = np.zeros((8, 10))
for j, p in enumerate(path1):
    value = [float(values[i]) for i, x in enumerate(paths) if x == str(p)]
    k[j, :] = np.array(value)






x = np.array([np.full((1, 10), 1, dtype=float), np.full((1, 10), 2, dtype=float), np.full((1, 10), 3, dtype=float), np.full((1, 10), 4, dtype=float),np.full((1, 10), 5, dtype=float),
              np.full((1, 10), 6, dtype=float),np.full((1, 10), 7, dtype=float),np.full((1, 10), 8, dtype=float)])
plt.scatter(x, k, color='b', marker='_')
plt.title("Sigmoid_output scatter")
plt.show()

