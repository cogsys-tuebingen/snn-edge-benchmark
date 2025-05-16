import sys
sys.path.append('../../dataset/')


import torch
import torch.nn as nn
import random
import pandas as pd
import numpy  as np
import sinabs
from sinabs.from_torch import from_model
from prepare_data import load_data



LEARNINGRATE = 0.0001

SNNBATCHSIZE = 200
TIMESTEPSSNNSIM = 8
INPUTSIZE = (1,64,64)

FRAMECOUNT_PROJ = 7569
FRAMECOUNT_HL = 2123

TESTRUN = True

data_folder_projected = 'dataset/projected/'
data_folder_handlabeled = 'dataset/hand_labeled/'


def trainTransformTest(data_folder, trained_network_file):

    torch.cuda.empty_cache() 
    torch.set_printoptions(precision=6)

    Model = nn.Sequential(
        nn.modules.conv.Conv2d(1, 4, 5,stride = 2, bias = False),	
        nn.ReLU(),
        nn.AvgPool2d(2,2),

        nn.modules.conv.Conv2d(4, 4, 3, bias = False),
        nn.ReLU(),
        nn.AvgPool2d(2,2),
        
        nn.modules.conv.Conv2d(4, 64, 6, stride = 6 , bias = False),
        nn.ReLU(),

        nn.modules.conv.Conv2d(64, 256, 1, bias = False),
        nn.ReLU(),
    ).to(torch.float32)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=LEARNINGRATE*5)

    # Projected data
    projected_frames, projected_positions = load_data(data_folder_projected)

    # Handlabeled data
    handlabeled_frames, handlabeled_positions = load_data(data_folder_handlabeled)

    basedata, baseTruth = torch.from_numpy(projected_frames), torch.from_numpy(projected_positions)
    data_hl, gt_hl = handlabeled_frames, handlabeled_positions

    train_hl = torch.from_numpy(data_hl[round((len(data_hl)*0.5)):round((len(data_hl)*1))])
    train_hl_gt = torch.from_numpy(gt_hl[round((len(data_hl)*0.5)):round((len(data_hl)*1))])

    test_hl = torch.from_numpy(data_hl[round((len(data_hl)*0)):round((len(data_hl)*0.25))])
    test_hl_gt = torch.from_numpy(gt_hl[round((len(data_hl)*0)):round((len(data_hl)*0.25))])

    basedata = torch.concat((basedata, train_hl))
    baseTruth = torch.concat((baseTruth, train_hl_gt))

    Trainbatches = [batch for batch in zip(basedata, baseTruth)]
    Testbatches = [batch for batch in zip(test_hl, test_hl_gt)]

    random.shuffle(Trainbatches)
                                   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model.to(device) 

    snnFinetuneReduceSynops(Model, loss_function, Trainbatches, Testbatches, trained_network_file = trained_network_file)

    Model = torch.load(trained_network_file)
    Model.to(device)

    print("final Performance SNN:")
    print(Model)

    print(test(Model, Testbatches))
 
    


def test(Model, testbatches):

    
    device = torch.device('cpu')
    Model.to(device)
        

        
    SynopsList = []
    Model.eval()
    batch_euclid_list = []

    with torch.no_grad():


        batch_size = SNNBATCHSIZE
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(testbatches):

            batch_in.append(batch[0].to(device))

            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(testbatches) - 1:
                batch_in = torch.stack(batch_in)

                
                Model.reset_states()
                if batch_idx == len(testbatches)-1:
                    break

                batch_in = torch.unsqueeze(batch_in, dim = 1)

                counter = sinabs.SNNSynOpCounter(Model.spiking_model)

                batch_in = torch.unsqueeze(batch_in, dim = 1)

                batch_in_new = batch_in
                for i in range(TIMESTEPSSNNSIM-1):
                    batch_in_new = torch.cat((batch_in_new,batch_in), dim = 1)
                
                batch_in = batch_in_new

                batch_in = sinabs.layers.FlattenTime()(batch_in).to(torch.float32)
                

                batch_gt = torch.stack(batch_gt)
                batch_gt = torch.squeeze(batch_gt)
 
                batch_out = Model(batch_in)
               
                if len(batch_out.size()) == 4:
                    batch_out = batch_out.unsqueeze(dim = 1)
    
                SynopsList.append(counter.get_total_synops())
                print(counter.get_synops())
            
                batch_out = batch_out.unflatten(
                    0, (batch_size,batch_out.shape[0] // batch_size)
                    )
                
                batch_out = batch_out.squeeze()
                
                batch_out = torch.mean(batch_out, 1) 

                if len(batch_gt.size()) == 1:
                    batch_gt = batch_gt.unsqueeze(dim = 0)
  
                if len(batch_out.size()) == 1:
                    batch_out = batch_out.unsqueeze(dim = 0)

                PredX = torch.argmax(batch_out[:,0:128],dim = 1)
                PredY = torch.argmax(batch_out[:,128:256],dim = 1)

                Model.reset_states()

                batch_euclid_list.append(torch.sqrt((PredX-batch_gt[:,0])**2 + (PredY-batch_gt[:,1])**2))


                del(batch_in)
                del(batch_gt)
                del(batch_out)
                batch_in = []
                batch_gt = []

    euclid = torch.cat(batch_euclid_list,dim = 0)
    print("mean euclidian distance")
    print(torch.mean(euclid))
    print("std of euclidian distance")
    print(torch.std(euclid))
    

    print("SYNOPS")
    print(SynopsList)
    print(torch.mean(torch.stack(SynopsList)))
    
    
    return (torch.mean(euclid), torch.std(euclid), torch.mean(torch.stack(SynopsList)), TIMESTEPSSNNSIM)
 


def snnFinetuneReduceSynops(Model,loss_function, Trainbatches, Testbatches, trained_network_file = 'a'):
    device = 'cuda'
    SNN = True
    torch.autograd.set_detect_anomaly((True))
    with torch.no_grad():
        snn = from_model(
            Model, input_shape=INPUTSIZE, add_spiking_output=False, min_v_mem = -1,synops=False, num_timesteps=TIMESTEPSSNNSIM,
            batch_size= SNNBATCHSIZE
        )
        snn.to(device)
        
        synops_counter = sinabs.SNNSynOpCounter(snn.spiking_model)
        
        calibrate_fac = 0.007/(90000*SNNBATCHSIZE)

    
        
        nr_epochs = 100
        param_layers = [name for name, child in snn.spiking_model.named_children() if isinstance(child, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))]
        named_layers = dict(snn.spiking_model.named_children())
        for i in range(len(param_layers)):
            param_layer = named_layers[param_layers[i]]
            param_layer.weight *= 2
    
    snn.train()
    optimizer_snn = torch.optim.Adam(snn.parameters(),lr=5*LEARNINGRATE)
    euclidmean, euclidstd, synopsmean, timesteps = test(snn, Testbatches)
    bestsofar = euclidmean

    for epoch in range(nr_epochs):
        synops_counter = sinabs.SNNSynOpCounter(snn.spiking_model)

        random.shuffle(Trainbatches)

        torch.cuda.empty_cache()

        

        batch_in = []
        batch_gt = []
        

        snn.to('cpu')
        euclidmean, euclidstd, synopsmean, timesteps = test(snn, Testbatches)
        if euclidmean < bestsofar:
            bestsofar = euclidmean
            torch.save(snn, trained_network_file)

        print(euclidmean)
        snn.to(device)

        for batch_idx, batch in enumerate(Trainbatches):

            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))
            
            
            if batch_idx == len(Trainbatches)-1:
                break

            if (batch_idx + 1) % SNNBATCHSIZE == 0 or batch_idx == len(Trainbatches) - 1:
                snn.reset_states()
                batch_in = torch.stack(batch_in)

                batch_in = torch.unsqueeze(batch_in, dim = 1)
                
                with torch.no_grad():
                    batch_in = torch.unsqueeze(batch_in, dim = 1)
                    batch_in_new = batch_in
                    for i in range(TIMESTEPSSNNSIM-1):
                        batch_in_new = torch.cat((batch_in_new,batch_in), dim = 1)
                    batch_in = batch_in_new
                    batch_in = sinabs.layers .FlattenTime()(batch_in).to(torch.float32)

                batch_gt = torch.stack(batch_gt)
                batch_gt = torch.squeeze(batch_gt)

                batch_out = snn(batch_in)
                batch_out = batch_out.squeeze().clone()
                

                batch_out = batch_out.unflatten(
                    0, (SNNBATCHSIZE,batch_out.shape[0] // SNNBATCHSIZE)
                    ).clone()
                batch_out = torch.mean(batch_out, 1).clone()


                batch_gtX = torch.round(batch_gt[:,0]).long()
                batch_gtY = torch.round(batch_gt[:,1]).long()
     
                target = torch.zeros(batch_out.size()).cuda()
                print(target.size())
                
                for i in range(batch_out.size()[0]):

                    target[i,batch_gtX[i]] = 1
                    target[i,batch_gtX[i]-1] = 0.5
                    target[i,batch_gtX[i]+1] = 0.5

                    target[i,128+batch_gtY[i]] = 1
                    target[i,128+batch_gtY[i]-1] = 0.5
                    target[i,128+batch_gtY[i]+1] = 0.5

                loss = loss_function(batch_out,target)
                
                MaxNorm = 0
                param_layers = [name for name, child in snn.spiking_model.named_children() if isinstance(child, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))]
                named_layers = dict(snn.spiking_model.named_children())
                for i in range(len(param_layers)):
                    param_layer = named_layers[param_layers[i]]
                    MaxNorm += torch.max(torch.square(param_layer.weight))

                synop_loss = torch.abs(synops_counter.get_total_synops())*calibrate_fac
                synops_counter.get_synops()


                loss = loss + synop_loss + 0.01 * MaxNorm

                optimizer_snn.zero_grad()
                
                loss.backward(retain_graph=True)
               
                optimizer_snn.step()
                

                del(batch_in)
                del(batch_gt)
                del(batch_out)
                batch_in = []
                batch_gt = []


        print(epoch)
    
    snn = torch.load(trained_network_file)
    print(snn)


    snn.eval()
    snn.to('cpu')

    return(snn)

