# import os
# import sys
# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# grandpaeentdir = os.path.dirname(parentdir)
# sys.path.insert(0, parentdir)
# sys.path.insert(0, grandpaeentdir)

import numpy as np
import cv2
import os
from ANN.libs.ImgDraw import *


if __name__ == '__main__':
    img_path = '/media/zcf/T7/slip-dataset_2.1/tactile_image_motion'  # path to tactile_image_motion
    posi_path = '/media/zcf/T7/slip-dataset_2.1/marker_position'  # path to marker_position

    save_path = '/media/zcf/T7/slip-dataset_2.1/tactile_image_motion_neighbor'  # path to save new image

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for obj in os.listdir(img_path):   # object
        print('--------- {:s}----------'.format(obj))
        obj_img_path = os.path.join(img_path, obj)
        obj_posi_path = os.path.join(posi_path, obj)

        obj_save_path = os.path.join(save_path, obj)

        for group in os.listdir(obj_img_path):   # group
            print(group)
            group_img_path = os.path.join(obj_img_path, group)
            group_posi_path = os.path.join(obj_posi_path, group)

            group_save_path = os.path.join(obj_save_path, group)
            if not os.path.exists(group_save_path):
                os.makedirs(group_save_path)

            num = len(os.listdir(group_img_path))

            for i in range(2, num+1):
                img = cv2.imread(group_img_path + '/' + str(i) + '.png')
                posi = np.load(group_posi_path + '/' + str(i) + '.npy')
                posi_last = np.load(group_posi_path + '/' + str(i-1) + '.npy')

                posi_L = posi[0]
                posi_R = posi[1]
                posi_last_L = posi_last[0]
                posi_last_R = posi_last[1]

                posi_L = np.transpose(posi_L, (1, 2, 0))
                posi_L = posi_L.reshape(-1, 2).tolist()

                posi_R = np.transpose(posi_R, (1, 2, 0))
                posi_R = posi_R.reshape(-1, 2)
                posi_R = posi_R + np.array([1280, 0])
                posi_R = posi_R.tolist()

                posi_last_L = np.transpose(posi_last_L, (1, 2, 0))
                posi_last_L = posi_last_L.reshape(-1, 2).tolist()

                posi_last_R = np.transpose(posi_last_R, (1, 2, 0))
                posi_last_R = posi_last_R.reshape(-1, 2)
                posi_last_R = posi_last_R + np.array([1280, 0])
                posi_last_R = posi_last_R.tolist()

                img = draw_arrow_2(img, posi_last_L+posi_last_R, posi_L+posi_R, color=(255, 0, 0))

                cv2.imwrite(group_save_path + '/' + str(i) + '.png', img)





