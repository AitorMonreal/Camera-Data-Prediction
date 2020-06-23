from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycpd.rigid_registration as rigid_registration
import numpy as np
import time
import open3d as o3d
import os


PCs = 5
# INITIALISE THE MODEL
pca_path = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/OAI-ZIB/processed_data/07.pca_tests/deformable'
pca_components = np.load(pca_path+'/pca_components.npy')[0:PCs, :]
pca_mean = np.load(pca_path+'/pca_mean.npy')

instance = np.asarray([200, 300, 90, 50, 40])
X_new = np.matmul(instance, np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
model_instance = np.zeros([int(X_new.size/3), 3])
#model_instance[:, 0] = X_new[0, 0:int(X_new.size/3)]
#model_instance[:, 1] = X_new[0, int(X_new.size/3):int(2*X_new.size/3)]
#model_instance[:, 2] = X_new[0, int(2*X_new.size/3):int(X_new.size)]

model_instance[:, 0] = X_new[0:int(len(X_new) / 3)]
model_instance[:, 1] = X_new[int(len(X_new) / 3):int(2 * len(X_new) / 3)]
model_instance[:, 2] = X_new[int(2 * len(X_new) / 3):len(X_new)]

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(model_instance[:, 0], model_instance[:, 1], model_instance[:, 2])
# plt.show()

#INITIAL TRIMMING
model_instance = model_instance[model_instance[:, 2].argsort(kind='mergesort')] #sorting row by third column (z)
model_instance = model_instance[:500, :]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(model_instance[:, 0], model_instance[:, 1], model_instance[:, 2])
plt.show()

np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Predictions/11/Test 2/model_instance_1_quarter.npy', model_instance)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(model_instance)
o3d.io.write_point_cloud('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Predictions/11/Test 2/model_instance_1_quarter.ply', pcd)
