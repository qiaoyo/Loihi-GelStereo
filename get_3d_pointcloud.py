import os
import numpy as np
from libs.BlobStereoMatching import *
from libs.config import *

if __name__ == '__main__':
    load_path = '/media/zcf/Documents/slip-dataset_2.1/marker_position'  # path of marker_position

    pointcloud_save_path = '/media/zcf/Documents/slip-dataset_2.1/3D_pointcloud'

    for obj in os.listdir(load_path):  # object
        obj_load_path = os.path.join(load_path, obj)
        pointcloud_obj_save_path = os.path.join(pointcloud_save_path, obj)
        print(obj)
        for group in os.listdir(obj_load_path):  # group
            print(group)
            group_load_path = os.path.join(obj_load_path, group)
            pointcloud_group_save_path = os.path.join(pointcloud_obj_save_path, group)

            if not os.path.exists(pointcloud_group_save_path):
                os.makedirs(pointcloud_group_save_path)

            num = len(os.listdir(group_load_path))
            for i in range(1, num + 1):
                marker = np.load(os.path.join(group_load_path, str(i) + '.npy'), allow_pickle=True)

                marker_L = marker[0]
                marker_R = marker[1]

                marker_sorted_L = np.transpose(marker_L, (1, 2, 0))
                marker_sorted_L = marker_sorted_L.reshape(-1, 2).tolist()
                marker_sorted_R = np.transpose(marker_R, (1, 2, 0))
                marker_sorted_R = marker_sorted_R.reshape(-1, 2).tolist()

                points = structure_blob_stereo_matching(marker_sorted_L, marker_sorted_R, Q, flag_sorting=False)

                np.save(os.path.join(pointcloud_group_save_path, str(i) + '.npy'), points)
        print('------------------------------')
