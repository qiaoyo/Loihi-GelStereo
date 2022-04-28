from datetime import datetime
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import torch
import random
import torch.optim
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
    if file_size % 4 != 0:
        print('corrupted event data file.', file_size)
        return None, 0

    with open(path, 'rb') as f:
        data = f.read(file_size)

    cnt = file_size // 4

    idx = 0
    start_ts = data[0] & 0x3fff

    xEvent, yEvent, pEvent, tEvent = [], [], [], []

    for i in range(cnt):
        datum=((data[i*4]<<24)|(data[i*4+1]<<16)|(data[i*4+2]<<8)|(data[i*4+3]))

        x = np.uint16(datum >> 24)
        y = np.uint16((datum >> 16) & 0xff)
        p = (datum >> 14) & 0x3
        t = np.uint32(datum & 0x3fff)

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
        self.samples=np.loadtxt(sampleFile,dtype=str)
        self.samplingTime=samplingTime
        self.nTimeBins=int(sampleLength/samplingTime)

    def __getitem__(self, index)  :
        inputIndex=self.samples[index,0]
        classLabel=int(self.samples[index,1])

        inputSpikes=read2Dspikes(os.path.join(self.path,inputIndex+'.bs2')).toSpikeTensor(torch.zeros((4,26,26,self.nTimeBins)))

        desiredClass=torch.zeros((2,1,1,1))
        desiredClass[classLabel,...]=1

        return inputSpikes,desiredClass,classLabel

    def __len__(self):
        return self.samples.shape[0]


##def the network
class Network(torch.nn.Module):
    def __init__(self,netParams):
        super(Network,self).__init__()
        slayer=snn.loihi(netParams['neuron'],netParams['simulation'])
        self.slayer=slayer
        self.conv1=slayer.conv(4,8,5,padding=1)
        self.conv2=slayer.conv(8,16,3,padding=1)
        self.conv3=slayer.conv(16,32,3,padding=1)
        self.pool1=slayer.pool(2)
        self.pool2=slayer.pool(2)
        self.pool3=slayer.pool(2)
        self.fc1 = slayer.dense((3,3,32),512)
        self.fc2 = slayer.dense(512,2)
        self.drop=slayer.dropout(0.1)

    def forward(self,spikeInput):
        spike=self.slayer.spikeLoihi(self.conv1(spikeInput)) #24,24,8
        spike=self.slayer.delayShift(spike,1)

        spike=self.slayer.spikeLoihi(self.pool1(spike)) #12,12,8
        spike=self.slayer.delayShift(spike,1)

        spike=self.drop(spike)
        spike=self.slayer.spikeLoihi(self.conv2(spike)) #12,12,16
        spike=self.slayer.delayShift(spike,1)

        spike=self.slayer.spikeLoihi(self.pool2(spike)) #6,6,16
        spike=self.slayer.delayShift(spike,1)

        spike=self.drop(spike)
        spike=self.slayer.spikeLoihi(self.conv3(spike)) #6,6,32
        spike=self.slayer.delayShift(spike,1)

        spike = self.slayer.spikeLoihi(self.pool3(spike))  # 3,3,32
        spike = self.slayer.delayShift(spike, 1)

        spike=self.drop(spike)
        spike = self.slayer.spikeLoihi(self.fc1(spike))
        spike=self.slayer.delayShift(spike,1)

        spike = self.slayer.spikeLoihi(self.fc2(spike))
        spike=self.slayer.delayShift(spike,1)

        return spike

if __name__ == "__main__"  :

    netParams=snn.params('network.yaml')
    device = torch.device('cuda')

    print(torch.cuda.is_available())
    f1=open('04007/evaluate.txt','w')
    f1.write('#precision #recall #f1_score\n')

    net=Network(netParams).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    # cudnn.benchmark = True
    # path='./0407/trained_gelstereo.pt'
    # net.load_state_dict(torch.load(path))


    error=snn.loss(netParams,snn.loihi).to(device)

    optimizer = snn.utils.optim.Nadam(net.parameters(), lr = 2e-3, amsgrad = True)

    trainingSet = gelstereoDataset(datasetPath=netParams['training']['path']['in'],
                                sampleFile=netParams['training']['path']['train'],
                                samplingTime=netParams['simulation']['Ts'],
                                sampleLength=netParams['simulation']['tSample'])
    trainLoader = DataLoader(dataset=trainingSet, batch_size=256, shuffle=False, num_workers=1)

    testingSet = gelstereoDataset(datasetPath=netParams['training']['path']['in'],
                               sampleFile=netParams['training']['path']['test'],
                               samplingTime=netParams['simulation']['Ts'],
                               sampleLength=netParams['simulation']['tSample'])
    testLoader = DataLoader(dataset=testingSet, batch_size=256, shuffle=False, num_workers=1)

    stats = snn.utils.stats()

    for epoch in range(200):
        tst=datetime.now()

        TP=0
        FP=0
        FN=0

        for i ,(input,target,label) in enumerate(trainLoader,0):
            net.train()
            input = input.to(device)
            print(input.shape)
            target = target.to(device)

            output=net.forward(input)

            #numSpikes = torch.sum(output, 4, keepdim=True).cpu()
            #print(numSpikes.reshape((numSpikes.shape[0], -1)))
            #output_class = snn.predict.getClass(output)
            #print(output_class, end=' label:')
            #print(label)

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
            output_class = snn.predict.getClass(output)

            if epoch%20==1:
                numSpikes = torch.sum(output, 4, keepdim=True).cpu()
                print(numSpikes.reshape((numSpikes.shape[0], -1)))
                print(output_class, end='label:')
                print(label)

            stats.testing.correctSamples+=torch.sum(output_class == label ).data.item()
            stats.testing.numSamples += len(label)

            for j in range(len(label)):
                if output_class[j]==1 and label[j]==1:
                    TP+=1
                if output_class[j]==1 and label[j]==0:
                    FP+=1
                if output_class[j]==0 and label[j]==1:
                    FN+=1

            loss = error.numSpikes(output,target)
            stats.testing.lossSum+=loss.cpu().data.item()
        if (TP+FP)!=0 and (TP+FN)!=0:
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            if precision!=0 or recall!=0:
                f1_score=2*precision*recall/(precision+recall)
                f1.write(str(precision)+"\t"+str(recall)+"\t"+str(f1_score)+"\n")
        else:
            f1.write("TP+FP==0 or TP+FN==0"+"\n")
        # if epoch%10 ==0: stats.print(epoch,timeElapsed=(datetime.now()-tst).total_seconds())
        stats.update()
        if stats.training.bestLoss is True: torch.save(net.state_dict(),'./04007/trained_gelstereo.pt')

    # plot the accuracy and loss,save file

    plt.semilogy(stats.training.lossLog,label='Training')
    plt.semilogy(stats.testing.lossLog,label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./04007/Loss.png')
    plt.clf()

    plt.plot(stats.training.accuracyLog,label='Training')
    plt.plot(stats.testing.accuracyLog,label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./04007/Accuracy.png')
    stats.save('04007/')

