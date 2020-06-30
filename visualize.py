# %%
import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np


# %%
data_dir = '/data1/pointnet_test/shapenetcore_partanno_segmentation_benchmark_v0'
_dlist = os.listdir(data_dir)
dlist = [x for x in _dlist if os.path.isdir(os.path.join(data_dir, x))]
dlist.remove('train_test_split')


# %%
def show_data(dir_num, file_num):
    dir_path = data_dir + '/' + dlist[dir_num] + '/points'
    file_list = os.listdir(dir_path)
    file_path = os.path.join(dir_path, file_list[file_num])

    pcd = o3d.io.read_point_cloud(file_path, format='xyz')
    print(pcd)

    label = np.zeros(len(pcd.points))
    max_label = label.max()
    colors = plt.get_cmap('tab20')(label / (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    o3d.visualization.draw_geometries([pcd], width=800, height=600)


# %%
show_data(0,0)