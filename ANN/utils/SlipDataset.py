import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2


def get_data(data_path, txt_path, seq_length):
    dir_list = []
    label_list = []
    motion_seq_list = []

    # read dir file
    with open(txt_path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.strip('\n').split('\t')
                dir_list.append(line[0])
                label_list.append(int(line[1]))
            else:
                break

    # read motion
    for i in range(len(dir_list)):
        dir_split = dir_list[i].split('_')
        file = int(dir_split[-1])
        group = dir_split[-2]
        if len(dir_split) == 5:
            dir = dir_split[0] + '_' + dir_split[1] + '_' + dir_split[2]
        elif len(dir_split) == 4:
            dir = dir_split[0] + '_' + dir_split[1]
        else:
            print('dir error')

        motion_seq = []
        for s in range(seq_length):
            # motion load path
            load_path = os.path.join(data_path, dir + '/' + group + '/' + str(file+1+s) + '.npy')
            motion = np.load(load_path)
            motion_seq.append(motion)

        motion_seq_list.append(motion_seq)
    return motion_seq_list, label_list



class SlipDataset(Dataset):
    def __init__(self, seq_length=3, phase='train'):
        txt_path = '/home/zcf/Documents/PyCharm project/gelstereo-slip-detection/dataset/' + phase + '.txt'  # path to train.txt or test.txt
        data_path = '/media/zcf/T7/slip-dataset_2.1/diff_motion'  # data path
        self.motion_seq_list, self.label_list = get_data(data_path=data_path, txt_path=txt_path, seq_length=seq_length)



    def __getitem__(self, index):
        motion_seq_temp = self.motion_seq_list[index]
        label_temp = self.label_list[index] # 1 slip 0 stable

        motion_seq = []
        for m in motion_seq_temp:
            motion_seq.append(torch.from_numpy(m).float())
        motion_seq = torch.stack(motion_seq)

        if label_temp == 0: # stable
            label = torch.Tensor([0, 1]).float()
        elif label_temp == 1:  # slip
            label =torch.Tensor([1, 0]).float()

        return motion_seq, label


    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    dataset = SlipDataset(seq_length=3, phase='train')

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 4

    )
    # batch_size = 16,
    # shuffle = True,
    # num_workers = 4
    for index, (ds, gt) in enumerate(data_loader):
        print(index)
        # print(ds)
        print(ds.shape)
        print(gt)

        # print(ds[0].shape)
        # print(ds[1].shape)

    # txt_path = '../dataset/test.txt'
    # data_path = '/media/zcf/T7/slip-dataset_2.1/diff_motion'
    #
    # motion_seq_list, label_list = get_data(data_path=data_path, txt_path=txt_path, seq_length=3)
    #
    # print(label_list[6])
    # if label_list[6] == 0:
    #     print('stalbe')
    # elif label_list[6] == 1:
    #     print('slip')