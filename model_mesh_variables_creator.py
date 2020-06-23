import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import open3d as o3d
import os
import trimesh
import pyglet

target_load = o3d.io.read_point_cloud('model.ply')
point_cloud = np.asarray(target_load.points)

meshlab_model_mesh = trimesh.load('model.stl')

vertices = np.asarray(meshlab_model_mesh.vertices)
faces = np.asarray(meshlab_model_mesh.faces)

indexes = []
for row in range(point_cloud.shape[0]):
    found = False
    for row2 in range(vertices.shape[0]):
        if row == row2:
            found = True
    if found == False:
        indexes.append(row)

indexes = np.asarray(indexes)
trimmed = np.delete(point_cloud, indexes, axis=0)

T = np.matmul(vertices, np.linalg.pinv(trimmed))

PCs = 5
# INITIALISE THE MODEL
pca_components = np.load(pca_components.npy')[0:PCs, :]
pca_mean = np.load(pca_mean.npy')

X_new = np.matmul(np.zeros([1, PCs]), np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
model = np.zeros([int(X_new.size / 3), 3])
model[:, 0] = X_new[0, 0:int(X_new.size / 3)]
model[:, 1] = X_new[0, int(X_new.size / 3):int(2 * X_new.size / 3)]
model[:, 2] = X_new[0, int(2 * X_new.size / 3):int(X_new.size)]

trimmed_model = np.delete(model, indexes, axis=0)

reordered_model = np.matmul(T, trimmed_model)

mesh = trimesh.Trimesh(vertices=reordered_model,
                       faces=faces)

np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Points_to_Surface/faces_connectivity_file.npy', faces)
np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Points_to_Surface/mesh_trimming_indexes.npy', indexes)
np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Points_to_Surface/reordering_mesh_transformation.npy', T)
mesh.show()
