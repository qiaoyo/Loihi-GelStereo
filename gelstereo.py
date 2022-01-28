from datetime import datetime
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn

from slayerSNN import learningStats as learningStats
from slayerSNN import optimizer as optimizer

class Event():

    def __init__(self, xEvent, yEvent, pEvent, tEvent):
        if yEvent is None:
            self.dim = 1
        else:
            self.dim = 2

        self.x = xEvent if type(xEvent) is np.ndarray else np.asarray(xEvent)  # x spatial dimension
        self.y = yEvent if type(yEvent) is np.ndarray else np.asarray(yEvent)  # y spatial dimension
        self.p = pEvent if type(pEvent) is np.ndarray else np.asarray(pEvent)  # spike polarity
        self.t = tEvent if type(tEvent) is np.ndarray else np.asarray(tEvent)  # time stamp in ms

        if not issubclass(self.x.dtype.type, np.integer): self.x = self.x.astype('int')
        if not issubclass(self.p.dtype.type, np.integer): self.p = self.p.astype('int')

        if self.dim == 2:
            if not issubclass(self.y.dtype.type, np.integer): self.y = self.y.astype('int')

        self.p -= self.p.min()

class snnio_update_loadbin(snn.io):
    def read2Dspikes(path):

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
        events = Event(xEvent, yEvent, pEvent, [time / 1000 for time in tEvent])

        print(type(events))

        return events

netParams=snn.params('network.yaml')

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

        inputSpikes=snnio_update_loadbin.read2Dspikes(self.path+str(inputIndex.item())+'.bs2').toSpikeTensor(torch.zeros((2,25,25,self.nTimeBins)))

        desiredClass=torch.zeros((2,1,1,1))
        desiredClass[classLabel,...]=1

        return inputSpikes.reshape((-1,1,1,inputSpikes.shape[-1])),desiredClass,classLabel

    def __len__(self):
        return self.samples.shape[0]


##def the network
class Network(torch.nn.Module):
    def __init__(self,netParams):
        super(Network,self).__init__()
        slayer=snn.layer(netParams['neuron'],netParams['simulation'])
        self.slayer=slayer

        self.fc1 = slayer.dense((25,25,2),512)
        self.fc2 = slayer.dense(512,2)

    def forward(self,spikeInput):
        spikeLayer1=self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeLayer2=self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))

        return spikeLayer2


device = torch.device('cuda')
net=Network(netParams).to(device)

error=snn.loss(netParams).to(device)

optimizer=torch.optim.Adam(net.parameters(),lr=0.01,amsgrad=True)

trainingSet = gelstereoDataset(datasetPath=netParams['training']['path']['in'],
                            sampleFile=netParams['training']['path']['train'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)

testingSet = gelstereoDataset(datasetPath=netParams['training']['path']['in'],
                           sampleFile=netParams['training']['path']['test'],
                           samplingTime=netParams['simulation']['Ts'],
                           sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=4)

stats=learningStats()

for epoch in range(100):
    tst=datetime.now()

    for i ,(input,output,label) in enumerate(trainLoader,0):
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


    for i ,(input,target,label) in enumerate(testLoader,0):
        input=input.to(device)
        target=target.to(device)

        output=net.forward(input)

        stats.testing.correctSamples+=torch.sum(snn.predict.getClass(output) == label ).data.item()
        stats.testing.numSamples += len(label)

        loss = error.numSpikes(output,target)
        stats.testing.lossSum+=loss.cpu().data.item()

    if epoch%10 ==0: stats.print(epoch,timeElapsed=(datetime.now()-tst).total_seconds())
    stats.update()
    if stats.training.bestLoss is True: torch.save(net.state_dict(),'trained_gelstereo.pt')

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

