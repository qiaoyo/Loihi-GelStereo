from datetime import datetime
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn

from slayerSNN.spikeFileIO import event

def read2Dspikes(path):

    '''

    :param path: bs2 file path
    :return: event(Class format) from slayerSNN.spikeFileIO
    '''

    file_size = os.path.getsize(path)
    if file_size == 0:
        print('empty event data file.')
        return None, 0
    if file_size % 8 != 0:
        print('corrupted event data file.', file_size)
        return None, 0

    with open(path, 'rb') as f:
        data = f.read(file_size)

    cnt = file_size // 8

    idx = 0
    start_ts = data[0] & 0x7FFFFFFF

    xEvent, yEvent, pEvent, tEvent = [], [], [], []

    for i in range(cnt):
        datum = (data[i * 8] << 56) | (data[i * 8 + 1] << 48) | (data[i * 8 + 2] << 40) | (
                    data[i * 8 + 3] << 32) | (data[i * 8 + 4] << 24) | (data[i * 8 + 5] << 16) | (
                            data[i * 8 + 6] << 8) | (data[i * 8 + 7])

        x = np.uint16(datum >> 48)
        y = np.uint16((datum >> 32) & 0xffff)
        p = (datum >> 31) & 0x1
        t = np.uint32(datum & 0x7FFFFFFF)

        xEvent.append(x)
        yEvent.append(y)
        pEvent.append(p)
        tEvent.append(t)
        idx += 1

    duration = data[-1] & 0xFFFFFFFF
    events = event(xEvent, yEvent, pEvent, [time  for time in tEvent])

    return events



## load dataset
class gelstereoDataset(Dataset):
    def __init__(self,datasetPath,sampleFile,samplingTime,sampleLength):
        self.path=datasetPath
        self.samples=np.loadtxt(sampleFile).astype('int')
        self.samplingTime=samplingTime
        self.nTimeBins=int(sampleLength/samplingTime)

    def __getitem__(self, index):
        inputIndex=self.samples[index,0]
        classLabel=self.samples[index,1]

        inputSpikes=read2Dspikes(os.path.join(self.path,str(inputIndex.item())+'.bs2')).toSpikeTensor(torch.zeros((2,800,800,self.nTimeBins)))

        desiredClass=torch.zeros((2,1,1,1))
        desiredClass[classLabel,...]=1

        return inputSpikes,desiredClass,classLabel

    def __len__(self):
        return self.samples.shape[0]


##def the network
class Network(torch.nn.Module):
    def __init__(self,netParams):
        super(Network,self).__init__()
        slayer=snn.layer(netParams['neuron'],netParams['simulation'])
        self.slayer=slayer
        self.conv1=slayer.conv(2,16,5,padding=2,weightScale=10)
        self.conv2=slayer.conv(16,32,3,padding=1,weightScale=20)
        self.pool1=slayer.pool(10)
        self.pool2=slayer.pool(2)
        self.pool3=slayer.pool(2)
        self.fc1 = slayer.dense((20,20,32),512)
        self.fc2 = slayer.dense(512,2)
        self.drop=slayer.dropout(0.1)

    def forward(self,spikeInput):
        spike=self.slayer.spike(self.pool1(spikeInput)) #80,80,2
        spike=self.drop(spike)
        spike=self.slayer.spike(self.conv1(spike)) #80,80,16
        spike=self.slayer.spike(self.pool2(spike)) #40,40,16
        spike=self.drop(spike)
        spike=self.slayer.spike(self.conv2(spike)) #40,40,32
        spike=self.slayer.spike(self.pool3(spike)) #20,20,32
        spike=self.drop(spike)

        spike=self.slayer.spike(self.slayer.psp(self.fc1(spike)))
        spike=self.slayer.spike(self.slayer.psp(self.fc2(spike)))

        return spike

if __name__ == "__main__"  :

    netParams=snn.params('network.yaml')

    print(torch.cuda.is_available())

    device = torch.device('cuda')
    net=Network(netParams).to(device)
    net=torch.nn.DataParallel(net,device_ids=[0,1,2,3])

    error=snn.loss(netParams).to(device)

    optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 0.01, amsgrad = True)

    trainingSet = gelstereoDataset(datasetPath=netParams['training']['path']['in'],
                                sampleFile=netParams['training']['path']['train'],
                                samplingTime=netParams['simulation']['Ts'],
                                sampleLength=netParams['simulation']['tSample'])
    trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=1)

    testingSet = gelstereoDataset(datasetPath=netParams['training']['path']['in'],
                               sampleFile=netParams['training']['path']['test'],
                               samplingTime=netParams['simulation']['Ts'],
                               sampleLength=netParams['simulation']['tSample'])
    testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=1)

    stats = snn.utils.stats()

    for epoch in range(50):
        tst=datetime.now()

        for i ,(input,target,label) in enumerate(trainLoader,0):
            net.train()
            input = input.to(device)
            target = target.to(device)

            output=net.forward(input)

            stats.training.correctSamples +=torch.sum(snn.predict.getClass(output) == label ).data.item()
            stats.training.numSamples +=len(label)

            loss=error.numSpikes(output,target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            stats.training.lossSum+= loss.cpu().data.item()
            stats.print(epoch, i, (datetime.now() - tst).total_seconds())

        for i ,(input,target,label) in enumerate(testLoader,0):
            net.eval()
            with torch.no_grad():
                input=input.to(device)
                target=target.to(device)

            output=net.forward(input)

            stats.testing.correctSamples+=torch.sum(snn.predict.getClass(output) == label ).data.item()
            stats.testing.numSamples += len(label)

            loss = error.numSpikes(output,target)
            stats.testing.lossSum+=loss.cpu().data.item()

        # if epoch%10 ==0: stats.print(epoch,timeElapsed=(datetime.now()-tst).total_seconds())
        stats.update()
        if stats.training.bestLoss is True: torch.save(net.state_dict(),'trained_gelstereo.pt')

    # plot the accuracy and loss,save file

    plt.semilogy(stats.training.lossLog,label='Training')
    plt.semilogy(stats.testing.lossLog,label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss.png')
    plt.clf()

    plt.plot(stats.training.accuracyLog,label='Training')
    plt.plot(stats.testing.accuracyLog,label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracy.png')
    stats.save('Trained/')

