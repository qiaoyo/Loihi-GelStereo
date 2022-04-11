import os,sys
import numpy as np
import shutil
import random
np.set_printoptions(threshold=np.inf)
load_path='/home/robot/data/gelstereo_data_0117/event_0407/'
save_path='/home/robot/data/gelstereo_data_0117/0407/'


if not os.path.exists(save_path):
    os.makedirs(save_path)

slide=1
not_slide=1
total_slide=0
total_not_slide=0

for noun in os.listdir(load_path):
    slide_num=0
    not_slide_num=0
    noun_load_path=os.path.join(load_path,noun)
    noun_save_path=os.path.join(save_path,noun)
    if not os.path.exists(noun_save_path):
        os.makedirs(noun_save_path)
    slide_path=os.path.join(noun_save_path,'slide')
    not_slide_path = os.path.join(noun_save_path, 'not_slide')

    os.makedirs(slide_path)
    os.makedirs(not_slide_path)
#
    for state in os.listdir(noun_load_path):
        state_load_path=os.path.join(noun_load_path,state)

        num=len(os.listdir(state_load_path))//2
        if int(state) in range(21):

            total_slide+=num
            for i in range(num):
                shutil.copyfile(os.path.join(state_load_path,str(i+1)+'.bs2'),os.path.join(slide_path,str(slide_num+1)+'.bs2'))
                slide_num+=1
        else:

            total_not_slide+=num
            for i in range(num):
                shutil.copyfile(os.path.join(state_load_path,str(i+1)+'.bs2'),os.path.join(not_slide_path,str(not_slide_num+1)+'.bs2'))
                not_slide_num+=1
    print(noun,slide_num,not_slide_num)

print(total_slide)
print(total_not_slide)

load_path='/home/robot/data/gelstereo_data_0117/0407/'
save_path='/home/robot/data/gelstereo_data_0117/gel_0407/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

f1=open(os.path.join(save_path,'train.txt'),'w')
f2=open(os.path.join(save_path,'test.txt'),'w')

f1.write("#sample #class\n")
f2.write("#sample #class\n")


slide=0
not_slide=0
total_slide=4105
total_not_slide=3023
slide_test=0
not_slide_test=0
train=0
test=0

for noun in os.listdir(load_path):
    noun_load_path=os.path.join(load_path,noun)
    slide_load_path=os.path.join(noun_load_path,'slide')
    not_slide_load_path=os.path.join(noun_load_path,'not_slide')

    num_slide=len(os.listdir(slide_load_path))
    num_not_slide=len(os.listdir(not_slide_load_path))

    select1=round(num_slide/total_slide*1500)
    select2=round(num_not_slide/total_not_slide*1500)

    test_selcet1=(select1*3)//10
    test_selcet2=(select2*3)//10

    index1=np.random.randint(1,num_slide+1,size=select1)
    index2=np.random.randint(1,num_not_slide+1,size=select2)


    print(select2,select1,noun)
    for i in range(select1-test_selcet1):
        size=os.path.getsize(os.path.join(slide_load_path,str(index1[i])+'.bs2'))
        if size > 1000:
            shutil.copyfile(os.path.join(slide_load_path,str(index1[i])+'.bs2'),os.path.join(save_path,str(train+1)+'.bs2'))
            f1.write(str(train+1)+'\t'+'1'+'\n')
            slide+=1
            train+=1
    for i in range(select2-test_selcet2):
        size = os.path.getsize(os.path.join(not_slide_load_path, str(index2[i]) + '.bs2'))
        if size > 1000:
            shutil.copyfile(os.path.join(not_slide_load_path,str(index2[i])+'.bs2'),os.path.join(save_path,str(train+1)+'.bs2'))
            f1.write(str(train+1)+'\t'+'0'+'\n')
            not_slide += 1
            train+=1

    for i in range(test_selcet1):
        size = os.path.getsize(os.path.join(slide_load_path, str(index1[select1 - i-1]) + '.bs2'))
        if size > 1000:
            shutil.copyfile(os.path.join(slide_load_path, str(index1[select1 - i-1]) + '.bs2'), os.path.join(save_path, str(30000+test + 1) + '.bs2'))
            f2.write(str(30000+test + 1) + '\t' + '1' + '\n')
            slide_test += 1
            test+=1

    for i in range(test_selcet2):
        size = os.path.getsize(os.path.join(not_slide_load_path, str(index2[select2 - i - 1]) + '.bs2'))
        if size > 1000:
            shutil.copyfile(os.path.join(not_slide_load_path, str(index2[select2 - i - 1]) + '.bs2'), os.path.join(save_path, str(30000+test + 1) + '.bs2'))
            f2.write(str(30000 + test + 1) + '\t' + '0' + '\n')
            not_slide_test += 1
            test += 1

print(train,test)
print(slide,not_slide)
print(slide_test,not_slide_test)
f1.close()
f2.close()

load_path='/home/robot/data/gelstereo_data_0117/gel_0407/'

f1=open(os.path.join(load_path,'train.txt'),'r')
f2=open(os.path.join(load_path,'test.txt'),'r')
f3=open(os.path.join(load_path,'Train.txt'),'w')
f4=open(os.path.join(load_path,'Test.txt'),'w')
f3.write("#sample #class\n")
f4.write("#sample #class\n")

read_1=f1.readlines()
read_2=f2.readlines()

list_test=[]
for line in read_2[1:]:
    li=line.split('\t')
    tmp=[]
    tmp.append(int(li[0]))
    tmp.append(int(li[1][0]))
    list_test.append(tmp)

list_test=np.array(list_test)

slide=[]
not_slide=[]
for i in range(len(list_test)):
    if list_test[i][1]==1:
        slide.append(list_test[i])
    else:
        not_slide.append(list_test[i])

slide=np.array(slide)
not_slide=np.array(not_slide)
print(slide.shape)
print(not_slide.shape)

for i in range(min(len(slide),len(not_slide))):
    f4.write(str(slide[i][0])+'\t'+str(slide[i][1])+'\n')
    f4.write(str(not_slide[i][0])+'\t'+str(not_slide[i][1])+'\n')



list_train=[]
for line in read_1[1:]:
    li=line.split('\t')
    tmp=[]
    tmp.append(int(li[0]))
    tmp.append(int(li[1][0]))
    list_train.append(tmp)

list_train=np.array(list_train)

slide=[]
not_slide=[]
for i in range(len(list_train)):
    if list_train[i][1]==1:
        slide.append(list_train[i])
    else:
        not_slide.append(list_train[i])

slide=np.array(slide)
not_slide=np.array(not_slide)
print(slide.shape)
print(not_slide.shape)

for i in range(min(len(slide),len(not_slide))):
    f3.write(str(slide[i][0])+'\t'+str(slide[i][1])+'\n')
    f3.write(str(not_slide[i][0])+'\t'+str(not_slide[i][1])+'\n')
