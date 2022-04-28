slip detection using SNN for GelStereo 1.0

1、环境配置

使用anaconda做python环境管理，我所使用的是3.9.7，3.8应该也问题不大。

可以用environment.yml方便的生成conda的虚拟环境。其中的prefix修改为自己电脑内的anaconda3路径。

`conda env create -f environment.yml`

2、数据预处理

frame2event.py : 从原始marker点位置插值生成dvs格式的event数据，并保存为bs2文件。bs2文件为二进制文件，格式在该脚本中，为8byte一个event，可以用最后的训练文件gelstereo.py中的一个函数read2Dspike读取。

frame2event_new.py：从原始的marker点位置通过泊松编码得到event数据，并保存为bs2文件。bs2文件为4byte一个event，可以用最后的训练文件中的一个函数read2Dspikes读取。

select_data_new.py : 从所有生成的bs2文件中选择数据，生成一个数据集，再随机划分为7：3的比例做训练和测试。

3、超参配置

network.yml : LOIHI模式下超参配置和数据路径

network_srm.yml : SRM模式下的超参配置和数据路径

LOIHI和SRM模式下的超参配置不相同。

4、训练和测试程序

gelstereo.py : 

SRM模型下的训练过程和测试过程。

gelstereoloihi.py : 

LOIHI模型下的训练过程和测试过程。

SRM和LOIHI是SLAYER框架下的2种模式，SRM中的权重，层间数据都是32位浮点数，而LOIHI下的权重是浮点数，层间数据是整数。SLAYER框架下，LOIHI模式的训练是为了能够直接部署在LOIHI硬件上，权重之后会经过量化为8位有符号整数。两种模式都可以仿真完成，而LOIHI可以部署在硬件上，SRM不能。因为数据类型的不同，导致模型的训练和推理方式都不同，识别能力也有差别。