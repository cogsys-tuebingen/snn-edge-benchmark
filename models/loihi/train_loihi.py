import numpy as np
import lava.lib.dl.slayer as slayer
import torch
import pandas as pd

from lava.utils.dataloader.mnist import MnistDataset
print(np.reshape(MnistDataset().images[1], (28,28)))
np.set_printoptions(linewidth=np.inf)
import h5py

from prepare_data import load_data

data_folder_projected = 'dataset/projected/'
data_folder_handlabeled = 'dataset/hand_labeled/'

NUMBEROFTIMEFRAMES = 7500
NUMBEROFTIMEFRAMES_HL = 2123

Timesteps_per_image = 20


neuronparams = {}
neuronparams["threshold"] = 1
#neuronparams['activation'] = torch.nn.functional.relu

#neuronparams_dense = {}
neuronparams["threshold"] = 0.1
neuronparams["threshold_step"] = 0.1
#neuronparams["current_decay"] = 1
#neuronparams["voltage_decay"] = 0.05
neuronparams["threshold_decay"] = 0.1
neuronparams["refractory_decay"] = 0.1

neuronparams_sd = {}
neuronparams_sd["threshold"] = 1

#neuronparams_cuba["current_decay"] = 1
#neuronparams_cuba["voltage_decay"] = 0#1#0.05
neuronparams_sd["shared_param"] = True#False
neuronparams_sd["scale"] = 64
neuronparams_sd["requires_grad"] = True
neuronparams_sd["scale_grad"] = 0.1

neuronparams_sd["activation"] = torch.nn.functional.relu

neuronparams_cuba = {}
neuronparams_cuba["threshold"] = 0.25#0.2#0.1

neuronparams_cuba["current_decay"] = 1##1
neuronparams_cuba["voltage_decay"] = 0.05#0.1 #0.15
neuronparams_cuba["shared_param"] = True#False
neuronparams_cuba["scale"] = 64
neuronparams_cuba["requires_grad"] = True
neuronparams_cuba["scale_grad"] = 1




neuronparams_cuba_inp = {}
neuronparams_cuba_inp["threshold"] = 0.1#0.1
neuronparams_cuba_inp["scale"] = 64

#neuronparams_cuba_inp["activation"] = torch.nn.functional.relu

neuronparams_cuba_inp["current_decay"] = 1
neuronparams_cuba_inp["voltage_decay"] = 0#1#0.05

def mycustomClassifier(output):
    output = output.squeeze()
    output = torch.mean(output, dim = 2)
    return output#torch.argmax(output, dim = 1)

def myCustomLoss(output = 0, target = 0):
    output = output.squeeze()
    x = torch.mean(output, dim = 2)
    loss_funct = torch.nn.MSELoss()
    loss = loss_funct(x, target)
    return loss


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__() 
        
        self.blocks = torch.nn.ModuleList([# sequential network blocks
                slayer.block.cuba.Input(neuronparams_cuba_inp),
                slayer.block.cuba.Conv(neuronparams_cuba,  1, 4, 5, weight_norm=True, delay=False, stride = (2,2), weight_scale = 1),#(2,2)),
                slayer.block.cuba.Conv(neuronparams_cuba, 4, 8, 5, weight_norm=True, delay = False,stride = (2,2), weight_scale = 1),#(2,2)),
                
                slayer.block.cuba.Conv(neuronparams_cuba, 8, 64, 13, stride = (13,13), weight_norm=True, delay=False, weight_scale = 1),
                slayer.block.cuba.Conv(neuronparams_cuba, 64, 256, 1, stride = (1,1), weight_norm=True, delay=False, weight_scale = 1),
            ])

    def forward(self, x):
        for block in self.blocks:
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
        return x

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        simulation = h.create_group('simulation')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


def test(assistant, data,gtset):
    batch_inp_list = []
    batch_gt_list = []

    for image in range(len(data)):
        

        input = torch.unsqueeze((data[image]), dim = 0).float()
        ground_truth = (gtset[image]).long()
        
        batch_inp_list.append(input)
        batch_gt_list.append(ground_truth)
        
        if image % batchSize == 0 and image > 0:
            input_batch = torch.stack(batch_inp_list, dim = 0)

            gt_batch = torch.stack(batch_gt_list, dim = 0)
            batch_gtX = torch.round(gt_batch[:,0]*1).long()
            batch_gtY = torch.round(gt_batch[:,1]*1).long()
            
            # Loss with MSE
            target = torch.zeros(gt_batch.size()[0],256)#.cuda()
            for i in range(gt_batch.size()[0]):
                target[i,batch_gtX[i]] = 1
                target[i,batch_gtX[i]-1] = 0.5
                target[i,batch_gtX[i]+1] = 0.5
                target[i,128+batch_gtY[i]] = 1
                target[i,128+batch_gtY[i]-1] = 0.5
                target[i,128+batch_gtY[i]+1] = 0.5


            output = assistant.test(slayer.utils.time.replicate(input_batch, Timesteps_per_image).cuda(), target.float().cuda())

            output = output.squeeze()
            output = torch.mean(output, dim = 2)

            PredX = torch.argmax(output[:,0:128],dim = 1) #+0.5
                    
            PredY = torch.argmax(output[:,128:256],dim = 1) #+0.5
            euclid_list.append(torch.sqrt((PredX.cpu()-torch.squeeze(gt_batch[:,0].cpu()))**2 + (PredY.cpu()-torch.squeeze(gt_batch[:,1].cpu()))**2))

            batch_inp_list = []
            batch_gt_list = []

    euclid = torch.cat(euclid_list,dim = 0)

    print(f"mean euclidian distance: {torch.mean(euclid)}")

    return torch.mean(euclid)
 

thenet = Network()
print(dir(thenet))
print(dir(thenet.blocks[1]))
print(thenet.blocks[0].weight)
print(thenet.blocks[1].synapse)
print(thenet.blocks[1].synapse.weight)
print(dir(thenet.blocks[1].synapse.weight))
func = thenet.blocks[1].synapse.weight.grad_fn

print(dir(thenet.blocks[1].synapse.weight))
import time

input = torch.unsqueeze(torch.tensor(np.reshape(MnistDataset().images[1], (28,28))),axis = 0).float()
print(input.shape)
ground_truth = torch.unsqueeze(torch.tensor(MnistDataset().labels[1]), dim = 0).long()
print(ground_truth)

# Projected data
projected_frames, projected_positions = load_data(data_folder_projected)
projected_frames, projected_positions = projected_frames[0:NUMBEROFTIMEFRAMES], projected_positions[0:NUMBEROFTIMEFRAMES]

# Handlabeled data
handlabeled_frames, handlabeled_positions = load_data(data_folder_handlabeled)
handlabeled_frames, handlabeled_positions = handlabeled_frames[0:NUMBEROFTIMEFRAMES_HL], handlabeled_positions[0:NUMBEROFTIMEFRAMES_HL]

traindata = torch.concat((projected_frames,(handlabeled_frames[round((len(handlabeled_frames)*0.5)):round((len(handlabeled_frames)*1))])))
trainTruth = torch.concat((projected_positions,(handlabeled_positions[round((len(handlabeled_frames)*0.5)):round((len(handlabeled_frames)*1))])))

testdata = handlabeled_frames[0:round((len(handlabeled_frames)*0.25))]
testTruth = handlabeled_positions[0:round((len(handlabeled_frames)*0.25))]


print(ground_truth.type())
print(input.type())
#print(dir(thenet))


optimizer = torch.optim.Adam(thenet.parameters(), lr = 0.001)


optimizer.zero_grad()

print(dir(thenet))
print(thenet.parameters)

assistant = slayer.utils.Assistant(thenet.cuda(),myCustomLoss, optimizer, stats = slayer.utils.stats.LearningStats(), classifier=mycustomClassifier)

print(dir(assistant))
print(dir(assistant.stats))
print(dir(assistant.stats.training))
print(assistant.stats.training.accuracy)
assistant.stats.print(0)
epochs = 200

batchSize = 100
bestsofar = 1.35
imageset = traindata
gtset = trainTruth

for epoch in range(epochs):

    batch_inp_list = []
    batch_gt_list = []

    euclid_list = []
    
    testres = test(assistant, testdata, testTruth) 
    if testres < bestsofar:
        bestsofar = testres
        thenet.export_hdf5('slayer_network.net')
        
    for image in range(len(imageset)):
        
        

        input = torch.unsqueeze((imageset[image]), dim = 0).float()
        ground_truth = (gtset[image]).long()
        
        batch_inp_list.append(input)
        batch_gt_list.append(ground_truth)
        
        if image % batchSize == 0 and image > 0:
            input_batch = torch.stack(batch_inp_list, dim = 0)

            gt_batch = torch.stack(batch_gt_list, dim = 0)

            batch_gtX = torch.round(gt_batch[:,0]*1).long()
            batch_gtY = torch.round(gt_batch[:,1]*1).long()

            
            #Loss with MSE
            target = torch.zeros(gt_batch.size()[0],256)#.cuda()
            for i in range(gt_batch.size()[0]):
                target[i,batch_gtX[i]] = 1
                target[i,batch_gtX[i]-1] = 0.5
                target[i,batch_gtX[i]+1] = 0.5
                target[i,128+batch_gtY[i]] = 1
                target[i,128+batch_gtY[i]-1] = 0.5
                target[i,128+batch_gtY[i]+1] = 0.5

            output = assistant.train(slayer.utils.time.replicate(input_batch, Timesteps_per_image).cuda(), target.float().cuda())
            output = output.squeeze()
            output = torch.mean(output, dim = 2)

            PredX = torch.argmax(output[:,0:128],dim = 1)
            PredY = torch.argmax(output[:,128:256],dim = 1)
            euclid_list.append(torch.sqrt((PredX.cpu()-torch.squeeze(gt_batch[:,0].cpu()))**2 + (PredY.cpu()-torch.squeeze(gt_batch[:,1].cpu()))**2))
 
            batch_inp_list = []
            batch_gt_list = []


    euclid = torch.cat(euclid_list,dim = 0)

    assistant.stats.print(epoch)
    assistant.stats.new_line()
    assistant.stats.update()     

thenet.cpu()
