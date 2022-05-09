import numpy as np
import os
import random


if __name__ == '__main__':
    data_path = '/media/zcf/T7/slip-dataset_2.1/diff_motion'

    test_obj = ['10_watermelon', '10_watermelon_2', '9_orange', '9_orange_2', '19_spanner_2', '20_gear_2']
    trainval_obj = ['14_hammer_2',    '1_cuboid',     '4_peach',    '7_starfruit',
                    '15_nut_2',       '1_kiwifruit',  '4_peach_2',  '7_starfruit_2',
                    '11_littleorange',    '16_screw_2',         '5_apple',    '8_mango',
                    '11_littleorange_2',  '17_cuboid_2',    '2_eggplant',   '5_apple_2',  '8_mango_2',
                    '12_greenapple_2',    '18_cylinder_2',  '3_banana',     '6_pear',
                    '13_lemon_2',            '3_banana_2',   '6_pear_2'   ]

    val_ratio = 0.2

    test_sample = []
    trainval_sample = []

    # test
    for obj in test_obj:
        obj_path = os.path.join(data_path, obj)
        for group in os.listdir(obj_path):
            group_path = os.path.join(obj_path, group)
            if int(group) <= 20:
                label = '1'
            else:
                label = '0'

            num = len(os.listdir(group_path)) - 2  # number of samples in this group
            for i in range(1, num+1):
                sample = {'name': obj + '_' + group + '_' + str(i), 'label': label}

                test_sample.append(sample)

    with open('test_0507.txt', 'at') as file:
        for slp in test_sample:
            file.write('%s\t' % slp['name'])
            file.write('%s\n' % slp['label'])

    # trainval
    for obj in trainval_obj:
        obj_path = os.path.join(data_path, obj)
        for group in os.listdir(obj_path):
            group_path = os.path.join(obj_path, group)
            if int(group) <= 20:
                label = '1'
            else:
                label = '0'

            num = len(os.listdir(group_path)) - 2  # number of samples in this group
            for i in range(1, num+1):
                sample = {'name': obj + '_' + group + '_' + str(i), 'label': label}

                trainval_sample.append(sample)


    random.seed(100)
    random.shuffle(trainval_sample)
    # print(trainval_sample)

    index = len(trainval_sample) * val_ratio
    index = int(index)

    with open('val_0507.txt', 'at') as file:
        for slp in trainval_sample[:index]:
            file.write('%s\t' % slp['name'])
            file.write('%s\n' % slp['label'])

    with open('train_0507.txt', 'at') as file:
        for slp in trainval_sample[index:]:
            file.write('%s\t' % slp['name'])
            file.write('%s\n' % slp['label'])



