# -*- coding: utf-8 -*-
# THIS ONE IS THE ONE THAT WORKS
"""
Created on Fri Dec  6 16:58:12 2019

@author: monre
"""
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycpd.rigid_registration as rigid_registration
import numpy as np
import time
import open3d as o3d
import os


# UP TO LINE 93 THE CODE WORKS WELL, THEN COMES THE OPTIMISATION

def main(femur_target, model):
    #reg = rigid_registration(**{'X': femur_target, 'Y': model})
    reg = rigid_registration(**{'X': model, 'Y': femur_target})  # register the femur surface from the Depth Camera to
    # the model, minimising the distance between the points of the femur surface and the model
    reg.register()
    dictionary = {"transformed_model": reg.register()[0],
                  "probability": reg.register()[2],
                  "rotation_matrix": reg.register()[3],
                  "translation": reg.register()[4]}
    return dictionary

predictions_path = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Predictions/11'
data_path = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Predictions/11'
femur_load = o3d.io.read_point_cloud(data_path + '/femur_target_400_11.ply')
femur_target = np.asarray(femur_load.points)

PCs = 5  # set the desired number of PC values
# INITIALISE THE MODEL
pca_path = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/OAI-ZIB/processed_data/07.pca_tests/deformable'
pca_components = np.load(pca_path+'/pca_components.npy')[0:PCs, :]
pca_mean = np.load(pca_path+'/pca_mean.npy')

X_new = np.matmul(np.zeros([1, PCs]), np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
model = np.zeros([int(X_new.size/3), 3])
model[:, 0] = X_new[0, 0:int(X_new.size/3)]
model[:, 1] = X_new[0, int(X_new.size/3):int(2*X_new.size/3)]
model[:, 2] = X_new[0, int(2*X_new.size/3):int(X_new.size)]

# TRIMMING
femur_target = femur_target[femur_target[:, 2].argsort(kind='mergesort')]
min_z = femur_target[-1, 2]+10
model = model[model[:, 2].argsort(kind='mergesort')] #sorting row by third column (z)
individual_max_z = model[-1, 2] #since the femur rows have been sorted, the maximum z will simply be the value at the bottom, column 2
while individual_max_z > (min_z):
    model = np.delete(model, -1, axis=0)
    individual_max_z = model[-1, 2]

length_model_trimmed = model.shape[0]

# INITIAL REGISTRATION
dictionary = main(femur_target, model)
femur_target = dictionary['transformed_model']
probability = dictionary['probability']
rotation_matrix = dictionary['rotation_matrix']
translation = dictionary['translation']

# INITIALISE THE MODEL AGAIN TO REDO TRIMMING BASED ON WHERE THE femur_target IS NOW
X_new = np.matmul(np.zeros([1, PCs]), np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
model = np.zeros([int(X_new.size/3), 3])
model[:, 0] = X_new[0, 0:int(X_new.size/3)]
model[:, 1] = X_new[0, int(X_new.size/3):int(2*X_new.size/3)]
model[:, 2] = X_new[0, int(2*X_new.size/3):int(X_new.size)]

# TRIMMING
femur_target = femur_target[femur_target[:, 2].argsort(kind='mergesort')]
min_z = femur_target[-1, 2]
model = model[model[:, 2].argsort(kind='mergesort')] #sorting row by third column (z)
individual_max_z = model[-1, 2] #since the femur rows have been sorted, the maximum z will simply be the value at the bottom, column 2
while individual_max_z > (min_z):
    model = np.delete(model, -1, axis=0)
    individual_max_z = model[-1, 2]

length_model_trimmed = model.shape[0]

# REDO ICP REGISTRATION
dictionary = main(femur_target, model)
femur_target = dictionary['transformed_model']
probability = dictionary['probability']
rotation_matrix = dictionary['rotation_matrix']
translation = dictionary['translation']



# UP TO HERE IT'S THE INITIAL ICP AND IT WORKS WELL

# INITIAL GUESS
x0 = -translation[0]  # negative since this is the translation for the femur_target been applied to the model
y0 = -translation[1]
z0 = -translation[2]
b0 = -np.arcsin(-rotation_matrix[2, 0])
a0 = -np.arcsin(rotation_matrix[2, 1]/np.cos(b0))
g0 = -np.arcsin(rotation_matrix[1, 0]/np.cos(b0))

a = np.asarray([x0, y0, z0, a0, b0, g0])
b = np.zeros([1, PCs])  # all zeros for the PC values
initial_guess = np.concatenate([a, b[0, :]])

# OBJECTIVE FUNCTION TO MINIMISE
def objective(params):
    x, y, z, a, b, g, PC0, PC1, PC2, PC3, PC4 = params

    # MODEL DEFORMATION
    X_new = np.matmul([PC0, PC1, PC2, PC3, PC4], np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
    # Create the model based on the chosen principal component values - convert it into a 3x1995 array
    model = np.zeros([int(len(X_new) / 3), 3])
    model[:, 0] = X_new[0:int(len(X_new) / 3)]
    model[:, 1] = X_new[int(len(X_new) / 3):int(2 * len(X_new) / 3)]
    model[:, 2] = X_new[int(2 * len(X_new) / 3):len(X_new)]

    # The model is not trimmed, it is the full femur model already with a good initial guess of location with respect /n
    # to the femur_target (from the Camera Data) due to the ICP between the trimmed model and the femur_target

    # MODEL TRANSFORMATION
    t = np.asarray([x, y, z])
    T = np.tile(t, (model.shape[0], 1))
    R = np.asarray([[np.cos(g)*np.cos(b), -np.sin(g)*np.cos(a)+np.cos(g)*np.sin(b)*np.sin(a), np.sin(g)*np.sin(a)+np.cos(g)*np.sin(b)*np.cos(a)],
                    [np.sin(g)*np.cos(b), np.cos(g)*np.cos(a)+np.sin(g)*np.sin(b)*np.sin(a), -np.cos(g)*np.sin(a)+np.sin(g)*np.sin(b)*np.cos(a)],
                    [-np.sin(b), np.cos(b)*np.sin(a), np.cos(b)*np.cos(a)]])
    transformed_model = np.dot(model, R) + T  # the model gets translated and rotated according to the chosen parameters

    # FIND MINIMUM DISTANCE - for each point in the femur_target, we find the smallest Euclidean distance to a point /n
    # on the transformed model. We then find the RMS sum of these distances. This is the value to minimise, hence in /n
    # theory minimising the distances between the points in the femur_target and their closest points in the model, /n
    # both through model translation, rotation, and deformation:
    x1 = femur_target.reshape(3, -1)  # converts femur_target into 3x205 - f1, f2, f3, f4, ... f54
    x2 = transformed_model.reshape(3, -1)  # converts transformed model into 3x1995 - m1, m2, m3, m4, ... m1995
    x3 = np.repeat(x1, x2.shape[1], axis=1)  # creates f1, f1, f1, ... f2, f2, f2, ... f3, f3, f3, ...
    x4 = np.tile(x2, x1.shape[1])  # creates m1, m2, m3, m4, ... m1, m2, m3, m4, ... m1, m2, m3, m4, ...
    distances = np.linalg.norm(x3.reshape(3, -1) - x4, axis=0).reshape(-1, transformed_model.shape[0])  # finds Euclidean distance for /n
    # each combination of f and m, and reshapes the matrix into [[(f1-m1), (f1-m2), (f1-m3), ...], [(f2-m1), (f2-m2), (f2-m3), ...], [(f3-m1), (f3-m2), (f3-m3), ...], ...]
    min_dist = np.diagonal(distances)  # find minimum distance for each femur point (f1, f2, f3, ... f205)
    error = np.power((np.sum(np.power(min_dist, 2)) / femur_target.shape[0]), 0.5)  # get the RMS of that distance
    return error  # - this is the distance to minimise

# MINIMISATION
from scipy.optimize import minimize
res = minimize(objective, initial_guess, tol=0.001)

# Different minimisation approaches
from scipy.optimize import basinhopping
#res = basinhopping(objective, initial_guess, niter_success=10)

from scipy.optimize import differential_evolution
#bounds = [(-200, 200), (-200, 200), (-200, 200), (-200, 200), (-200, 200)]
#res = differential_evolution(objective, bounds=bounds, tol=0.01)

print(res)

# SAVING THE RESULTS
os.chdir(predictions_path)
final_params = res.x
iterations = res.nit
final_error = res.fun
final_time = time.time() - start_time

np.save('params_11_' + str(PCs) + '.npy', final_params)
np.save('iterations_11_' + str(PCs) + '.npy', iterations)
np.save('fun_11_' + str(PCs) + '.npy', final_error)
np.save('time_11_' + str(PCs) + '.npy', final_time)
