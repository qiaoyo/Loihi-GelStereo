import numpy as np
import os,sys
import torch
import math
np.set_printoptions(threshold=np.inf,linewidth=np.inf)

def poisson(datum,time,dt=1):
    shape, size = datum.shape, datum.numel()
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (time / dt)
    # generate poisson train from rate
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0
    # train to spike
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]
    spikes = spikes.T
    return spikes.tolist()


def relative(data):
    data_out=[]
    for i in range(len(data)-1):
        temp=data[i+1][0]-data[i][0]
        data_out.append(temp)
    return np.array(data_out)

def channel_2(data):
    output = []
    for i in range(len(data)):
        loc = np.where(data[i] < 0)
        loc = np.array(loc)
        empty = np.zeros((2, 25, 25))
        for j in range(loc.shape[1]):
            empty[loc[0][j]][loc[1][j]][loc[2][j]] = -data[i][loc[0][j]][loc[1][j]][loc[2][j]]
            data[i][loc[0][j]][loc[1][j]][loc[2][j]] = 0

        output.append(np.concatenate((data[i],empty),axis=0))
    output=np.array(output).astype(int)
    return torch.from_numpy(output)

def totrain(data,frame_T,slice_T):
    empty=torch.zeros((data.shape[0],4,25,25,frame_T))
    Train=[]

    slice_num=math.ceil(slice_T/frame_T)

    for i in range(data.shape[0]):
        temp1=[]
        for j in range(data.shape[1]):
            temp2=[]
            for m in range(data.shape[2]):
                spikes=poisson(data[i][j][m],frame_T)
                temp2.append(spikes)
            temp1.append(temp2)
        Train.append(temp1)

    Train=torch.Tensor(Train)
    Train=Train[0:data.shape[0]]
    print(Train.shape)
    slice_Train=[]
    for i in range(data.shape[0]-2):
        temp=torch.cat((Train[i],Train[i+1],Train[i+2]),3)
        slice_Train.append(temp.numpy().tolist())


    return torch.tensor(slice_Train)

def saveTrain(data,path,noun,state):
    for i in range(data.shape[0]):
        file=open(path+'/'+noun+'_'+state+'_'+str(i+1)+'.bs2','wb')
        loc=torch.where(data[i]!=0)
        channel,x,y,t=loc
        print(len(channel),end=' ')
        event=[]
        if len(channel)!=0:
            for j in range(len(channel)):
                temp_event=[x[j],y[j],channel[j],t[j]]
                event.append(temp_event)
            event=np.array(event)
            xevent = event[:, 0]
            yevent = event[:, 1]
            cevent = event[:, 2]
            tevent = event[:, 3]

            outputByte = bytearray(len(xevent) * 4)

            outputByte[0::4] = np.uint8(xevent).tobytes()
            outputByte[1::4] = np.uint8(yevent).tobytes()
            outputByte[2::4] = np.uint8(((tevent>>8)&0x3f)|(cevent<<6)).tobytes()
            outputByte[3::4] = np.uint8(tevent&0xff).tobytes()

            file.write(outputByte)
    print()


if __name__=="__main__":

    load_path = '/home/robot/data/gelstereo_data_0117/marker_position'
    save_path = '/home/robot/data/gelstereo_data_0117/event_0427'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for noun in os.listdir(load_path):
        noun_load_path=os.path.join(load_path,noun)
        noun_save_path=os.path.join(save_path,noun)
        print(noun)
        if not os.path.exists(noun_save_path):
            os.makedirs(noun_save_path)
        for state in os.listdir(noun_load_path):
            # print(state,end=' ')
            state_load_path=os.path.join(noun_load_path,state)
            state_save_path=os.path.join(noun_save_path,state)
            if not os.path.exists(state_save_path):
                os.makedirs(state_save_path)

            data=[]
            num=len(os.listdir(state_load_path))
            for i in range(num):
                tmp=np.load(os.path.join(state_load_path,str(i+1)+'.npy'),'r')
                data.append(tmp)
            data=np.array(data)
            print(data.shape)
            print(state)
            # get relative x
            relative_x_list = relative(data)
            print(relative_x_list.shape)
            relative_x_chnnel2 = channel_2(relative_x_list)
            print(relative_x_chnnel2.shape)
            # to train
            Train=totrain(relative_x_chnnel2,43,100)
            print(Train.shape)
            saveTrain(Train,state_save_path,noun,state)
