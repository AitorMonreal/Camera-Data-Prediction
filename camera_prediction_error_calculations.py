import numpy as np
import open3d as o3d
import os
import shutil

downsampled = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/VCG/MeshLab/03.downsampled_PCDs'

results = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/Depth_Camera_Data/Points_to_Surface'
real_load = o3d.io.read_point_cloud(results + '/transformed_femur_target.ply')
femur_real = np.asarray(real_load.points)

error = 0
for filename in os.listdir(results):
    if filename == 'femur_prediction_PP_bounded_1_quarter_model.ply':
        prediction_load = o3d.io.read_point_cloud(results + '/' + filename)
        prediction = np.asarray(prediction_load.points)
        x1 = np.transpose(prediction)  # converts femur_target into 3x54 - f1, f2, f3, f4, ... f54
        x2 = np.transpose(femur_real)  # converts femur_target into 3x1995 - m1, m2, m3, m4, ... m1995
        x3 = np.repeat(x1, x2.shape[1], axis=1)  # creates f1, f1, f1, ... f2, f2, f2, ... f3, f3, f3, ...
        x4 = np.tile(x2, x1.shape[1])  # creates m1, m2, m3, m4, ... m1, m2, m3, m4, ... m1, m2, m3, m4, ...
        distances = np.linalg.norm(np.transpose(x3) - np.transpose(x4), axis=1).reshape(1995, -1)  # finds normal distance for each
        # combination of f and m, and reshapes the matrix into [[(f1-m1), (f1-m2), (f1-m3), ...], [(f2-m1), (f2-m2), (f2-m3), ...], [(f3-m1), (f3-m2), (f3-m3), ...], ...]
        min_dist = np.min(distances, axis=1)
        min_dist2 = np.min(distances, axis=0)
        error = np.power((np.sum(np.power(min_dist2, 2)) / femur_real.shape[0]), 0.5) #RMSD Error

        np.save(results + '/' + filename + '_RMSD_error.npy', np.array(error))
        print(filename)
        print(error)


print('done')
