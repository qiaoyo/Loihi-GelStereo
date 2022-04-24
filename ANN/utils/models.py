import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter


class ConvLSTM(nn.Module):
    def __init__(self, conv_1, conv_2, fc_1, lstm_layers, lstm_hidden_size, cls_num):
        super(ConvLSTM, self).__init__()
        self.CONV = nn.Sequential(nn.Conv2d(2, conv_1, (3, 3), stride=1),
                                  nn.BatchNorm2d(conv_1),
                                  nn.ReLU(True),
                                  nn.Dropout(p=0.1),
                                  nn.AvgPool2d(kernel_size=(2, 2)),
                                  nn.Conv2d(conv_1, conv_2, (3, 3), stride=1),
                                  nn.BatchNorm2d(conv_2),
                                  nn.ReLU(True),
                                  nn.Dropout(p=0.1),
                                  nn.AvgPool2d(kernel_size=(2, 2))
                                  )

        self.FC_feature = nn.Sequential(nn.Linear(conv_2 * 4 * 4, fc_1),
                                        nn.BatchNorm1d(fc_1),
                                        nn.Sigmoid(),
                                        nn.Dropout(p=0.1))

        self.LSTM = nn.LSTM(input_size=fc_1, hidden_size=lstm_hidden_size, num_layers=lstm_layers)

        self.FC_cls = nn.Sequential(nn.Linear(lstm_hidden_size, cls_num),
                                    nn.Sigmoid(),
                                    nn.Softmax(dim=1))

    def forward(self, motion_seq):
        '''
        :param motion_seq: (batch size, seq_length, 2, 25, 25)
        :return:
        '''
        motion_seq = motion_seq.permute(1, 0, 2, 3, 4) # (seq_length, batch size, 2, 25, 25)
        feature = []
        for i in range(motion_seq.shape[0]):
            motion = motion_seq[i]
            conv_f = self.CONV(motion)
            conv_f = conv_f.reshape(motion_seq.shape[1], -1)
            fc_f = self.FC_feature(conv_f)
            feature.append(fc_f)

        feature = torch.stack(feature, dim=0)  # (seq_length, batch_size, fc_1)

        LSTM_out, (h_n, h_c) = self.LSTM(feature, None)
        output = self.FC_cls(LSTM_out[-1, :, :])  # (batch size, fc_1)

        return output


class Conv(nn.Module):
    def __init__(self, in_channel, conv_1, conv_2, cls_num):
        super(Conv, self).__init__()
        self.CONV = nn.Sequential(nn.Conv2d(in_channel, conv_1, (3, 3), stride=1),
                                  nn.BatchNorm2d(conv_1),
                                  nn.ReLU(True),
                                  nn.Dropout(p=0.1),
                                  nn.AvgPool2d(kernel_size=(2, 2)),
                                  nn.Conv2d(conv_1, conv_2, (3, 3), stride=1),
                                  nn.BatchNorm2d(conv_2),
                                  nn.ReLU(True),
                                  nn.Dropout(p=0.1),
                                  nn.AvgPool2d(kernel_size=(2, 2))
                                  )

        self.FC_cls = nn.Sequential(nn.Linear(conv_2 * 4 * 4, cls_num),
                                    nn.Sigmoid(),
                                    nn.Softmax(dim=1))


    def forward(self, motion_seq):
        '''
        :param motion_seq: (batch size, seq_length, 2, 25, 25)
        :return:
        '''

        motion = motion_seq.reshape(motion_seq.shape[0], -1, motion_seq.shape[3], motion_seq.shape[4])

        feature = self.CONV(motion)
        output = self.FC_cls(feature.reshape(motion_seq.shape[0], -1))

        return output









