
Slip Detection Using SNN for GelStereo 1.0

1、Environment Configuration

Use Anaconda for Python environment management. The version I used is 3.9.7, and 3.8 should also work fine.

You can conveniently create a conda virtual environment using environment.yml. Modify the prefix in it to the path of anaconda3 on your own computer.

`conda env create -f environment.yml`

2、Data Preprocessing

frame2event.py: Interpolates from original marker positions to generate event data in DVS format and saves it as a bs2 file. The bs2 file is a binary file, with the format specified in this script (8 bytes per event), and can be read using the read2Dspike function in the final training file gelstereo.py.

frame2event_new.py: Obtains event data from original marker positions through Poisson encoding and saves it as a bs2 file. The bs2 file uses 4 bytes per event and can be read using the read2Dspikes function in the final training file.

select_data_new.py: Selects data from all generated bs2 files to create a dataset, which is then randomly split into a 7:3 ratio for training and testing.

3、Hyperparameter Configuration

network.yml: Hyperparameter configuration and data path for LOIHI mode

network_srm.yml: Hyperparameter configuration and data path for SRM mode

Hyperparameter configurations differ between LOIHI and SRM modes.

4、Training and Testing Programs

gelstereo.py:

Training and testing processes under the SRM model.

gelstereoloihi.py:

Training and testing processes under the LOIHI model.

SRM and LOIHI are two modes under the SLAYER framework. In SRM, weights and inter-layer data are all 32-bit floating-point numbers, while in LOIHI, weights are floating-point numbers and inter-layer data are integers. Under the SLAYER framework, training in LOIHI mode is intended for direct deployment on LOIHI hardware, with weights subsequently quantized to 8-bit signed integers. Both modes can be simulated, but only LOIHI can be deployed on hardware, while SRM cannot. Due to differences in data types, the model training and inference methods differ, as do their recognition capabilities.
