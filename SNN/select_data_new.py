import os,sys
import numpy as np
import shutil
import random
import random
np.set_printoptions(threshold=np.inf)
load_path='/home/robot/data/gelstereo_data_0117/event_0427/'
save_path='/home/robot/data/gelstereo_data_0117/0427/'


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

    for state in os.listdir(noun_load_path):
        state_load_path=os.path.join(noun_load_path,state)

        num=len(os.listdir(state_load_path))
        if int(state) in range(21):
            total_slide+=num
            for i in range(num):
                shutil.copyfile(os.path.join(state_load_path,noun+'_'+state+'_'+str(i+1)+'.bs2'),os.path.join(slide_path,noun+'_'+state+'_'+str(i+1)+'.bs2'))
                slide_num+=1
        else:
            total_not_slide+=num
            for i in range(num):
                shutil.copyfile(os.path.join(state_load_path,noun+'_'+state+'_'+str(i+1)+'.bs2'),os.path.join(not_slide_path,noun+'_'+state+'_'+str(i+1)+'.bs2'))
                not_slide_num+=1
    print(noun,slide_num,not_slide_num)

print(total_slide)
print(total_not_slide)

load_path='/home/robot/data/gelstereo_data_0117/0427/'
save_path='/home/robot/data/gelstereo_data_0117/gel_0427/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

f1=open(os.path.join(save_path,'train.txt'),'w')
f2=open(os.path.join(save_path,'test.txt'),'w')
f_delete=open(os.path.join(save_path,'delete.txt'),'w')

f1.write("#sample #class\n")
f2.write("#sample #class\n")


slide=0
not_slide=0
total_slide=24697
total_not_slide=18176
slide_test=0
not_slide_test=0
train=0
test=0
num_slice_min=0
size_slide_min=400
size_not_slide_min=40
delete_slide=0
delete_not_slide=0
for noun in os.listdir(load_path):
    noun_load_path=os.path.join(load_path,noun)
    slide_load_path=os.path.join(noun_load_path,'slide')
    not_slide_load_path=os.path.join(noun_load_path,'not_slide')

    num_slide=len(os.listdir(slide_load_path))
    num_not_slide=len(os.listdir(not_slide_load_path))

    select1=num_slide
    select2=num_not_slide

    test_select1=(select1*3)//10
    test_select2=(select2*3)//10

    index1=np.random.randint(1,num_slide+1,size=test_select1)
    index2=np.random.randint(1,num_not_slide+1,size=test_select2)

    slide_file=os.listdir(slide_load_path)
    not_slide_file=os.listdir(not_slide_load_path)
    random.shuffle(slide_file)
    random.shuffle(not_slide_file)

    for i in range(select1-test_select1):
        size=os.path.getsize(os.path.join(slide_load_path,slide_file[i]))
        if size > size_slide_min:
            shutil.copyfile(os.path.join(slide_load_path,slide_file[i]),os.path.join(save_path,slide_file[i]))
            f1.write(slide_file[i][:-4]+'\t'+'1'+'\n')
            slide+=1
            train+=1
        else:
            f_delete.write(slide_file[i][:-4]+'\t'+'1'+'\n')
            delete_slide+=1
            num_slice_min+=1
    for i in range(select2-test_select2):
        size = os.path.getsize(os.path.join(not_slide_load_path, not_slide_file[i]))
        if size > size_not_slide_min:
            shutil.copyfile(os.path.join(not_slide_load_path,not_slide_file[i]),os.path.join(save_path,not_slide_file[i]))
            f1.write(not_slide_file[i][:-4]+'\t'+'0'+'\n')
            not_slide += 1
            train+=1
        else:
            f_delete.write(not_slide_file[i][:-4]+'\t'+'0'+'\n')
            delete_not_slide+=1
            num_slice_min+=1
    for i in range(test_select1):
        size = os.path.getsize(os.path.join(slide_load_path, slide_file[select1-i-1]))
        if size > size_slide_min:
            shutil.copyfile(os.path.join(slide_load_path, slide_file[select1-i-1]), os.path.join(save_path, slide_file[select1-i-1]))
            f2.write(slide_file[select1-i-1][:-4] + '\t' + '1' + '\n')
            slide_test += 1
            test+=1
        else:
            f_delete.write(slide_file[select1-i-1][:-4] + '\t' + '1' + '\n')
            delete_slide+=1
            num_slice_min+=1
    for i in range(test_select2):
        size = os.path.getsize(os.path.join(not_slide_load_path, not_slide_file[select2-i-1]))
        if size > size_not_slide_min:
            shutil.copyfile(os.path.join(not_slide_load_path, not_slide_file[select2-i-1]), os.path.join(save_path, not_slide_file[select2-i-1]))
            f2.write(not_slide_file[select2-i-1][:-4] + '\t' + '0' + '\n')
            not_slide_test += 1
            test += 1
        else:
            f_delete.write(not_slide_file[select2-i-1][:-4] + '\t' + '0' + '\n')
            delete_not_slide+=1
            num_slice_min+=1

f_delete.write(str(delete_slide)+' '+str(delete_not_slide))
print(train,test)     #28798 12316
print(slide,not_slide)
print(slide_test,not_slide_test)
print(num_slice_min)
print(delete_slide,delete_not_slide)
f1.close()
f2.close()

load_path='/home/robot/data/gelstereo_data_0117/gel_0427'

f1=open(os.path.join(load_path,'train.txt'),'r')
f2=open(os.path.join(load_path,'test.txt'),'r')
f3=open(os.path.join(load_path,'train_reorder.txt'),'w')
f4=open(os.path.join(load_path,'test_reorder.txt'),'w')
f3.write("#sample #class\n")
f4.write("#sample #class\n")

read_1=f1.readlines()
read_2=f2.readlines()

list_test=[]
for line in read_2[1:]:
    li=line.split('\t')
    tmp=[]
    tmp.append(li[0])
    tmp.append(int(li[1][0]))
    list_test.append(tmp)

slide=[]
not_slide=[]
for i in range(len(list_test)):
    if list_test[i][1]==1:
        slide.append(list_test[i])
    else:
        not_slide.append(list_test[i])

print(len(slide))
print(len(not_slide))

for i in range(len(not_slide)):
    f4.write(slide[i][0]+'\t'+str(slide[i][1])+'\n')
    f4.write(not_slide[i][0]+'\t'+str(not_slide[i][1])+'\n')

list_train=[]
for line in read_1[1:]:
    li=line.split('\t')
    tmp=[]
    tmp.append(li[0])
    tmp.append(int(li[1][0]))
    list_train.append(tmp)

slide=[]
not_slide=[]
for i in range(len(list_train)):
    if list_train[i][1]==1:
        slide.append(list_train[i])
    else:
        not_slide.append(list_train[i])

print(len(slide))
print(len(not_slide))

for i in range(min(len(slide),len(not_slide))):
    f3.write(slide[i][0]+'\t'+str(slide[i][1])+'\n')
    f3.write(not_slide[i][0]+'\t'+str(not_slide[i][1])+'\n')


