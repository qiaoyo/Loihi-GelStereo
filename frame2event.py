import numpy as np
import os,sys

def interpolation(data,interpolation):
    '''

    :param data: input frame data
    :param interpolation: the num of frame to be interpolated between the original marker_position
    :return: the frame data .The format and content is identical with the marker position
    '''
    rst=[]
    for i in range(data.shape[0]-1):
        rst.append(data[i])
        for j in range(interpolation):
            new_img=data[i]+np.around((data[i+1]-data[i]) * (j + 1) / (interpolation + 1))
            rst.append(new_img)
    rst.append(data[data.shape[0]-1])
    return np.array(rst).astype(int)

def np2bin(data,state_save_path):

    # the event in 10 frames ,about 109ms ---> 'left'.bs2 and 'right'.bs2
    '''
                                    8 Bytes
    \--------\--------\--------\--------\--------\--------\--------\--------\
    \        x        \        y        \p\                t                \
    range of x:
        left: about 100~750
        right:about 750~1400
    range of y:
        about 100~750
    p:
        1 for appear
        0 for disappear
    range of t:
        (1,109)
    :param data: input frame data
    :param state_save_path: the path to save bs2 file
    :return: none
    '''

    bs2num=data.shape[0]//10

    for bs2name in range(bs2num):
        f1 = open(state_save_path + '/'+ str(bs2name+1) + 'right.bs2', 'wb')
        f2 = open(state_save_path + '/'+ str(bs2name+1) + 'left.bs2', 'wb')

        for i in range(bs2name*10, bs2name*10+10):
            if i != 0:
                event_left = []
                event_right = []
                tmp = data[i] - data[i - 1]
                loc = np.where(tmp != 0)
                loc = np.array(loc)
                temp=i % 10 + 1
                if loc.shape[1]!=0:
                    for j in range(loc.shape[1]):

                        if loc[0][j]==0:
                            tmp_event = [data[i][0][0][loc[2][j]][loc[3][j]], data[i][0][1][loc[2][j]][loc[3][j]], 1,
                                         np.around(temp * 10.869).astype(int)]
                            if tmp_event not in event_right:
                                event_right.append(tmp_event)
                                tmp_event = [data[i - 1][0][0][loc[2][j]][loc[3][j]], data[i - 1][0][1][loc[2][j]][loc[3][j]], 0,
                                         np.around(temp * 10.869).astype(int)]
                                event_right.append(tmp_event)
                        else :
                            tmp_event = [data[i][1][0][loc[2][j]][loc[3][j]], data[i][1][1][loc[2][j]][loc[3][j]], 1,
                                         np.around(temp * 10.869).astype(int)]
                            if tmp_event not in event_left:
                                event_left.append(tmp_event)
                                tmp_event = [data[i - 1][1][0][loc[2][j]][loc[3][j]],data[i - 1][1][1][loc[2][j]][loc[3][j]], 0,
                                         np.around(temp * 10.869).astype(int)]
                                event_left.append(tmp_event)


                    event_left = np.array(event_left)
                    event_right = np.array(event_right)
                    # print(event_right.shape)
                    # print(event_left.shape)


                    if len(event_left)!=0:
                        print(f"the min x:{min(event_left[:, 0])},the max x:{max(event_left[:, 0])}", end=' ')
                        print(f"the min y:{min(event_left[:, 1])},the max y:{max(event_left[:, 1])}")
                        outputByte = bytearray(len(event_left) * 8)
                        xevent=event_left[:,0]
                        yevent=event_left[:,1]
                        pevent = event_left[:, 2]
                        tevent = event_left[:, 3]

                        outputByte[0::8] = np.uint8(xevent >> 8).tobytes()
                        outputByte[1::8] = np.uint8(xevent & 0xff).tobytes()
                        outputByte[2::8] = np.uint8(yevent >> 8).tobytes()
                        outputByte[3::8] = np.uint8(yevent & 0xff).tobytes()

                        outputByte[4::8] = np.uint8(((tevent >> 24) & 0x7f) | (pevent << 7)).tobytes()
                        outputByte[5::8] = np.uint8((tevent >> 16) & 0xff).tobytes()
                        outputByte[6::8] = np.uint8((tevent >> 8) & 0xff).tobytes()
                        outputByte[7::8] = np.uint8(tevent & 0xff).tobytes()


                        f2.write(outputByte)
                    if len(event_right)!=0:

                        outputByte = bytearray(len(event_right) * 8)
                        xevent = event_right[:, 0]
                        yevent = event_right[:, 1]
                        pevent = event_right[:, 2]
                        tevent = event_right[:, 3]

                        outputByte[0::8] = np.uint8(xevent >> 8).tobytes()
                        outputByte[1::8] = np.uint8(xevent & 0xff).tobytes()
                        outputByte[2::8] = np.uint8(yevent >> 8).tobytes()
                        outputByte[3::8] = np.uint8(yevent & 0xff).tobytes()

                        outputByte[4::8] = np.uint8(((tevent >> 24) & 0x7f) | (pevent << 7)).tobytes()
                        outputByte[5::8] = np.uint8((tevent >> 16) & 0xff).tobytes()
                        outputByte[6::8] = np.uint8((tevent >> 8) & 0xff).tobytes()
                        outputByte[7::8] = np.uint8(tevent & 0xff).tobytes()

                        f1.write(outputByte)
        f1.close()
        f2.close()
    print()

if __name__=="__main__":
    load_path = './marker_position'
    save_path = './gelstereo_event'
    # load_path = '/home/robot/data/dataset/gelstereo_data_0117/marker_position'
    # save_path = '/home/robot/data/dataset/gelstereo_data_0117/gelstereo_event'
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
            # print("shape:{}".format(np.array(data).shape),end=' ')

            ## interpolation 3 and return npy. The format and content is identical with the marker position
            data=interpolation(data,3)
            print(data.shape)
            ## transform the frame npy to event.bs2. the bs2 is encoded with special format.
            np2bin(data,state_save_path)



