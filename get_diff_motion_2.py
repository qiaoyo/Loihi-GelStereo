import os
import numpy as np

if __name__ == '__main__':
    load_path = '/media/zcf/Documents/slip-dataset_2.1/marker_position'  # path of marker_position

    motion_save_path = '/media/zcf/Documents/slip-dataset_2.1/diff_motion'

    for obj in os.listdir(load_path):  # object
        obj_load_path = os.path.join(load_path, obj)
        motion_obj_save_path = os.path.join(motion_save_path, obj)
        print(obj)
        for group in os.listdir(obj_load_path):  # group
            print(group)
            group_load_path = os.path.join(obj_load_path, group)
            motion_group_save_path = os.path.join(motion_obj_save_path, group)

            if not os.path.exists(motion_group_save_path):
                os.makedirs(motion_group_save_path)

            num = len(os.listdir(group_load_path))
            for i in range(2, num+1):
                marker = np.load(os.path.join(group_load_path, str(i)+'.npy'), allow_pickle=True)
                marker_pre = np.load(os.path.join(group_load_path, str(i-1) + '.npy'), allow_pickle=True)

                marker_L = marker[0]
                marker_pre_L = marker_pre[0]

                motion = marker_L - marker_pre_L

                np.save(os.path.join(motion_group_save_path, str(i)+'.npy'), motion)
        print('------------------------------')


