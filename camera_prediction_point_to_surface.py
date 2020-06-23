from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycpd.rigid_registration as rigid_registration
import numpy as np
import time
import open3d as o3d
import os
import trimesh

start_time = time.time()


def main(f_model, f_femur_target):
    # reg = rigid_registration(**{'X': femur_target, 'Y': model})
    reg = rigid_registration(**{'X': f_model, 'Y': f_femur_target})  # register the femur surface from the Depth Camera to
    # the model, minimising the distance between the points of the femur surface and the model
    reg.register()
    f_dictionary = {"transformed_model": reg.register()[0],
                    "probability": reg.register()[2],
                    "rotation_matrix": reg.register()[3],
                    "translation": reg.register()[4]}
    return f_dictionary


femur_target = np.load('model_instance.npy')

PCs = 5
# INITIALISE THE SURFACE MODEL
pca_components = np.load('pca_components.npy')[0:PCs, :]
pca_mean = np.load('pca_mean.npy')

X_new = np.matmul(np.zeros([1, PCs]), np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
model = np.zeros([int(X_new.size / 3), 3])
model[:, 0] = X_new[0, 0:int(X_new.size / 3)]
model[:, 1] = X_new[0, int(X_new.size / 3):int(2 * X_new.size / 3)]
model[:, 2] = X_new[0, int(2 * X_new.size / 3):int(X_new.size)]

faces = np.load('faces_connectivity_file.npy')
indexes = np.load('mesh_trimming_indexes.npy')
T = np.load('reordering_mesh_transformation.npy')

trimmed_model = np.delete(model, indexes, axis=0)
reordered_model = np.matmul(T, trimmed_model)

mesh = trimesh.Trimesh(vertices=reordered_model,
                       faces=faces)
model_vertices = np.asarray(mesh.vertices)

#a = trimesh.proximity.closest_point(mesh, femur_target)[1]
# b = trimesh.proximity.signed_distance(mesh, femur_target)
# c = trimesh.proximity.nearby_faces(mesh, femur_target)
# nearby_faces = np.asarray([np.asarray(xi)[0] for xi in c]).reshape((-1, 1))  # an index of the nearby face to each point

# REGISTRATION GUESS
dictionary = main(model_vertices, femur_target)
#femur_target = dictionary['transformed_model']
probability = dictionary['probability']
rotation_matrix = dictionary['rotation_matrix']
translation = dictionary['translation']

#INITIAL GUESS
x0 = translation[0]
y0 = translation[1]
z0 = translation[2]
b0 = np.arcsin(-rotation_matrix[2, 0])
a0 = np.arcsin(rotation_matrix[2, 1]/np.cos(b0))
g0 = np.arcsin(rotation_matrix[1, 0]/np.cos(b0))

a = np.asarray([x0, y0, z0, a0, b0, g0])
b = np.zeros([1, PCs])
initial_guess = np.concatenate([a, b[0, :]])

# MODEL TRANSFORMATION
t0 = np.asarray([x0, y0, z0])
Tr0 = np.tile(t0, (model.shape[0], 1))  # Translation matrix
R0 = np.asarray([[np.cos(g0) * np.cos(b0), -np.sin(g0) * np.cos(a0) + np.cos(g0) * np.sin(b0) * np.sin(a0),
                 np.sin(g0) * np.sin(a0) + np.cos(g0) * np.sin(b0) * np.cos(a0)],
                [np.sin(g0) * np.cos(b0), np.cos(g0) * np.cos(a0) + np.sin(g0) * np.sin(b0) * np.sin(a0),
                 -np.cos(g0) * np.sin(a0) + np.sin(g0) * np.sin(b0) * np.cos(a0)],
                [-np.sin(b0), np.cos(b0) * np.sin(a0), np.cos(b0) * np.cos(a0)]])  # Rotation matrix
model_initial_guess = np.dot(model, R0) + Tr0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(model_initial_guess)
o3d.io.write_point_cloud('model_initial_guess.ply', pcd)

# OBJECTIVE FUNCTION TO MINIMISE
def objective(f_params, f_pca_components, f_pca_mean, f_faces, f_indexes, f_T, f_femur_target):
    f_x, f_y, f_z, f_a, f_b, f_g, f_PC0, f_PC1, f_PC2, f_PC3, f_PC4 = f_params
    # MODEL DEFORMATION
    f_X_new = np.matmul([f_PC0, f_PC1, f_PC2, f_PC3, f_PC4], np.transpose(np.linalg.pinv(f_pca_components))) + np.transpose(f_pca_mean)
    f_model = np.zeros([int(len(f_X_new) / 3), 3])
    f_model[:, 0] = f_X_new[0:int(len(f_X_new) / 3)]
    f_model[:, 1] = f_X_new[int(len(f_X_new) / 3):int(2 * len(f_X_new) / 3)]
    f_model[:, 2] = f_X_new[int(2 * len(f_X_new) / 3):len(f_X_new)]

    f_trimmed_model = np.delete(f_model, f_indexes, axis=0)
    f_reordered_model = np.matmul(f_T, f_trimmed_model)

    #MODEL TRANSFORMATION
    f_t = np.asarray([f_x, f_y, f_z])
    f_Tr = np.tile(f_t, (f_reordered_model.shape[0], 1)) #Translation matrix
    f_R = np.asarray([[np.cos(f_g)*np.cos(f_b), -np.sin(f_g)*np.cos(f_a)+np.cos(f_g)*np.sin(f_b)*np.sin(f_a), np.sin(f_g)*np.sin(f_a)+np.cos(f_g)*np.sin(f_b)*np.cos(f_a)],
                    [np.sin(f_g)*np.cos(f_b), np.cos(f_g)*np.cos(f_a)+np.sin(f_g)*np.sin(f_b)*np.sin(f_a), -np.cos(f_g)*np.sin(f_a)+np.sin(f_g)*np.sin(f_b)*np.cos(f_a)],
                    [-np.sin(f_b), np.cos(f_b)*np.sin(f_a), np.cos(f_b)*np.cos(f_a)]]) #Rotation matrix
    f_transformed_model = np.dot(f_reordered_model, f_R) + f_Tr

    f_mesh = trimesh.Trimesh(vertices=f_transformed_model,
                             faces=f_faces)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(transformed_model[:, 0], transformed_model[:, 1], transformed_model[:, 2])
    ax.scatter(femur_target[:, 0], femur_target[:, 1], femur_target[:, 2])
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    '''

    # POINT-TO-POINT ON SURFACE CALCULATION APPROACH
    # min_dist = trimesh.proximity.closest_point(mesh, femur_target)[1]

    # POINT-TO-SURFACE DISTANCE CALCULATION APPROACH
    triangle_indexes = trimesh.proximity.closest_point(f_mesh, f_femur_target)[2]  # 400x1 array with the indexes of the closest triangle to each point in femur_target
    closest_triangles = f_mesh.triangles[triangle_indexes]  # 400x3x3 array with the coordinates of the closest triangles. Each 3x3 are the coordinates of 3 points. Each row is a point/vertex making up the triangle
    point_on_triangles = closest_triangles[:, 0, :]  # the first point on each of the closest triangles
    vector_triangle_to_point = f_femur_target - point_on_triangles
    face_normals = f_mesh.face_normals[triangle_indexes]
    projection_vector_to_normal = vector_triangle_to_point * face_normals  # performs row-wise dot product between the two arrays, multiplying each element together
    min_dist = np.linalg.norm(projection_vector_to_normal, axis=1).reshape((-1, 1))

    error = np.power((np.sum(np.power(min_dist, 2)) / f_femur_target.shape[0]), 0.5)
    return error


# MINIMISATION
from scipy.optimize import minimize
res = minimize(fun=objective, x0=initial_guess, args=(pca_components, pca_mean, faces, indexes, T, femur_target), method='Powell', tol=0.01)
print(res)

# SAVING RESULTS
final_params = res.x
iterations = res.nit
final_error = res.fun
final_time = time.time() - start_time

np.save('params_11_' + str(PCs) + '.npy', final_params)
np.save('iterations_11_' + str(PCs) + '.npy', iterations)
np.save('fun_11_' + str(PCs) + '.npy', final_error)
np.save('time_11_' + str(PCs) + '.npy', final_time)

x = final_params[0]
y = final_params[1]
z = final_params[2]
b = final_params[3]
a = final_params[4]
g = final_params[5]
PC0 = final_params[6]
PC1 = final_params[7]
PC2 = final_params[8]
PC3 = final_params[9]
PC4 = final_params[10]
# PC5 = final_params[5]

X_new = np.matmul([PC0, PC1, PC2, PC3, PC4], np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
model = np.zeros([int(len(X_new) / 3), 3])
model[:, 0] = X_new[0:int(len(X_new) / 3)]
model[:, 1] = X_new[int(len(X_new) / 3):int(2 * len(X_new) / 3)]
model[:, 2] = X_new[int(2 * len(X_new) / 3):len(X_new)]

# MODEL TRANSFORMATION
t = np.asarray([x, y, z])
Tr = np.tile(t, (model.shape[0], 1))
R = np.asarray([[np.cos(g)*np.cos(b), -np.sin(g)*np.cos(a)+np.cos(g)*np.sin(b)*np.sin(a), np.sin(g)*np.sin(a)+np.cos(g)*np.sin(b)*np.cos(a)],
                [np.sin(g)*np.cos(b), np.cos(g)*np.cos(a)+np.sin(g)*np.sin(b)*np.sin(a), -np.cos(g)*np.sin(a)+np.sin(g)*np.sin(b)*np.cos(a)],
                [-np.sin(b), np.cos(b)*np.sin(a), np.cos(b)*np.cos(a)]])
transformed_model = np.dot(model, R) + Tr

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(transformed_model)
o3d.io.write_point_cloud('femur_prediction.ply', pcd)


print("--- %s seconds ---" % (time.time() - start_time))
print(final_time)
