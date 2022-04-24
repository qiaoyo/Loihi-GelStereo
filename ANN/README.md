slip detection using ANN for GelStereo 1.0

1、环境配置：

如果有anaconda可以先创建一个环境 python 3.8

requirement:
>numpy                         1.20.3
>torch                         1.10.0+cu113
>scikit-learn                  1.0.1
>matplotlib                    3.4.1

安装pytorch 1.10.0，根据CUDA Version选择对应版本
在https://pytorch.org/找到对应版本，复制指令安装即可

2、数据集

`SlipDataset.py`:需修改数据路径

>txt_path = '/home/zcf/Documents/PyCharm project/gelstereo-slip-detection/dataset/' + phase + '.txt'  # path to train.txt or test.txt <br />
>data_path = '/media/zcf/T7/slip-dataset_2.1/diff_motion'  # data path

data_path 为 相邻两帧之间marker position的差值，比如 2.npy表示marker_position/2.npy - marker_position/1.npy
get_diff_motion.py 为产生diff_motion的程序

3、训练模型 train.py

>--epochs 训练轮次<br />
>--test 每训练一轮进行测试<br />
>--use_cuda 使用GPU<br />
>--checkpoint 训练结果保存在该文件夹<br />
>--model_arch 选择模型<br />
>--resume 从该模型继续训练

例如：
>python train.py --epochs=50 --lr=0.01 --test --use_cuda --checkpoint=results/checkpoint_8 --model_arch=Conv

4、测试模型 test.py

>--use_cuda 使用GPU<br />
>--checkpoint 被测试模型所在文件夹<br />
>--model_arch 选择模型<br />
>--test_model 被测试模型<br />
>--dataset 测试数据集 testset/trainset

例如：
>python test.py --use_cuda --dataset=testset --checkpoint=results/checkpoint_8 --model_arch=Conv --test_model=model_best.pth.tar



