import numpy as np
import os,sys
import shutil
import random
load_path = './gelstereo_event'
save_path = './gelstereo'
cnt_train=1
cnt_test=1
f1=open(os.path.join(save_path,'train.txt'),'w')
f2=open(os.path.join(save_path,'test.txt'),'w')
f1.write("#sample #class\n")
f2.write("#sample #class\n")
for noun in os.listdir(load_path):
    noun_load_path=os.path.join(load_path,noun)

    for state in os.listdir(noun_load_path):
        state_load_path=os.path.join(noun_load_path,state)
        slide_num=len(os.listdir(state_load_path))//2  #36
        select_num=slide_num//9 #3 for train 1 for test
        select=np.random.randint(1,slide_num+1,size=select_num+1)

        for i in range(select_num):
            size=os.path.getsize(os.path.join(state_load_path,str(select[i])+'left.bs2'))
            if size>1000:
                shutil.copyfile(os.path.join(state_load_path,str(select[i])+'left.bs2'),os.path.join(save_path,str(cnt_train)+'.bs2'))
                f1.write(str(cnt_train))

                f1.write("\t")
                if int(state) <=20:
                    stat=1
                else :
                    stat=0
                f1.write(str(stat))
                f1.write("\n")
            print(str(cnt_train)+'left.bs2',state,stat,cnt_train)
            cnt_train += 1

        size = os.path.getsize(os.path.join(state_load_path, str(select[-1]) + 'left.bs2'))
        if size >1000:
            shutil.copyfile(os.path.join(state_load_path, str(select[-1]) + 'left.bs2'),os.path.join(save_path, '600'+str(cnt_test) + '.bs2'))
            f2.write('600'+str(cnt_test))

            f2.write("\t")
            if int(state) <=20:
                stat=1
            else :
                stat=0
            f2.write(str(stat))
            f2.write("\n")
            print('600'+str(cnt_test)+ 'left.bs2', state, stat, cnt_test)
            cnt_test += 1

