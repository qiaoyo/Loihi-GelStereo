
Slip Detection Using ANN for GelStereo 1.0

1、Environment Configuration:

If you have Anaconda, you can first create an environment with Python 3.8

Requirement:

> numpy                         1.20.3
> torch                         1.10.0+cu113
> scikit-learn                  1.0.1
> matplotlib                    3.4.1

Install PyTorch 1.10.0, and select the corresponding version according to the CUDA Version
Find the corresponding version on https://pytorch.org/, copy the command, and install it

2、Dataset

`SlipDataset.py`: Need to modify the data path

> txt_path = '/home/zcf/Documents/PyCharm project/gelstereo-slip-detection/dataset/' + phase + '.txt'  # path to train.txt or test.txt `<br />`
> data_path = '/media/zcf/T7/slip-dataset_2.1/diff_motion'  # data path

data_path is the difference in marker positions between adjacent frames. For example, 2.npy represents marker_position/2.npy - marker_position/1.npy
get_diff_motion.py is the program to generate diff_motion

3、Training the Model train.py

> --epochs Number of training epochs `<br />`
> --test Test after each training epoch `<br />`
> --use_cuda Use GPU `<br />`
> --checkpoint Folder to save training results `<br />`
> --model_arch Select model architecture `<br />`
> --resume Resume training from this model

Example:

> python train.py --epochs=50 --lr=0.01 --test --use_cuda --checkpoint=results/checkpoint_8 --model_arch=Conv

4、Testing the Model test.py

> --use_cuda Use GPU `<br />`
> --checkpoint Folder where the tested model is located `<br />`
> --model_arch Select model architecture `<br />`
> --test_model The model to be tested `<br />`
> --dataset Test dataset (testset/trainset)

Example:

> python test.py --use_cuda --dataset=testset --checkpoint=results/checkpoint_8 --model_arch=Conv --test_model=model_best.pth.tar
>
