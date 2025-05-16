import sys
sys.path.append('../../dataset/')

import torch
import time
import pandas as pd
import numpy  as np

import sinabs

from sinabs.backend.dynapcnn.io import  enable_timestamps
    
from sinabs.backend.dynapcnn import DynapcnnNetwork

from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from datetime import datetime
from typing import List
from prepare_data import load_data


NUMBEROFTIMEFRAMES = 2123
SNNBATCHSIZE = 1
TIMESTEPSSNNSIM = 1
TIMESTEPSONCHIP = 1

INPUTSIZE = (1,64,64)

data_folder_handlabeled = 'dataset/hand_labeled/'


def trainTransformTest(trained_network_file, onChip = False, runNum = 666):
    """
    Parameters:
        trained_network_file: Str;                   specifies the file location that a trained network can be loaded from and saved to
        onChip:               bool;                  should the model be used on the chip
    """

    # Should free up unused GPU memory, it is usualy more effective to start a new kernel
    torch.cuda.empty_cache() 
    torch.set_printoptions(precision=6)

    # Handlabeled data
    handlabeled_frames, handlabeled_positions = load_data(data_folder_handlabeled)

    data_hl, gt_hl = handlabeled_frames, handlabeled_positions

    test_hl = data_hl[round((len(data_hl)*0.25)):round((len(data_hl)*0.5))]
    test_hl_gt = gt_hl[round((len(data_hl)*0.25)):round((len(data_hl)*0.5))]
    
    Testbatches = [batch for batch in zip(test_hl, test_hl_gt)]        

    Model = torch.load(trained_network_file).to(torch.float32)
    Model.to('cuda')

    #SNN is saved with fixed batch size from training to determine speed per forward pass we adjust batch size to 1
    Model.spiking_model[1].num_timesteps = 1
    Model.spiking_model[4].num_timesteps = 1
    Model.spiking_model[7].num_timesteps = 1
    Model.spiking_model[9].num_timesteps = 1
    Model.spiking_model[1].batch_size = 1
    Model.spiking_model[4].batch_size = 1
    Model.spiking_model[7].batch_size = 1
    Model.spiking_model[9].batch_size = 1
    
    # to account for the lack of timesteps we multiply the input weights with 8
    Model.spiking_model[0].weight = torch.nn.Parameter(8 * Model.spiking_model[0].weight)
    print(Model.spiking_model[0].weight)
    Model.analog_model[0].weight = torch.nn.Parameter(8 * Model.analog_model[0].weight)
    
    print(Model.spiking_model[1].num_timesteps)
    time.sleep(5)
    
    if not(onChip):
        print(test(Model, Testbatches, runNum = runNum))
    
    if onChip:
        np_output_chip = test_DynapCNN(Model, Testbatches)
        np.savetxt('chip_output'+str(runNum)+'.csv', np_output_chip, delimiter = ',')



def convert_to_DynapCNN(SNNModel):
    """
    Converts a given Sinabs SNN into a DynapCNN-model which is the model type that can be employed on chip

    parameters:
        SNNmodel:   sinabs SNN;    the SNNModel
        inputsize: (int,int,int);  input dimensions of the Network
    """
    print(type(SNNModel.spiking_model[1]))
    SNNModel.to(torch.device('cpu'))
    print(SNNModel)
    DynapCNNModel = DynapcnnNetwork(
    SNNModel,
    discretize=True,
    input_shape=INPUTSIZE)
    print(DynapCNNModel.memory_summary())
    return DynapCNNModel

def test(Model, testbatches, runNum = 666):
    """
    The function for testing the ANN and testing the SNNs in simulation. Returns the loss for the testset.

    parameters:
        Model:         torch.nn.Sequential or Sinabs SNN; the model it tests
        testbatches:   torch.tensor;                      the data to test on
        runNum         int;                               which name to save
    """

    #selects the device since SNN simulation requires a lot of memory I run it on cpu
    
    device = torch.device('cpu')
    synops_counter = sinabs.SNNSynOpCounter(Model.spiking_model)
    Model.to(device)
        

    SynopsList = []
    Model.eval()
    start = True
    batch_diff_list = []
    batch_time_list = []
    batch_euclid_list = []
    
    # prevents gradient tracking, saves memory space 
    with torch.no_grad():
        # Initialize Loss
        total_loss = 0
        # Set Batchsize must match Batchsize of the SNN
        batch_size = SNNBATCHSIZE
        batch_in = []
        batch_gt = []
        # go trough all input ground truth pairs
        for batch_idx, batch in enumerate(testbatches):
        
            batch_in.append(torch.from_numpy(batch[0]).to(device))
      
            
            batch_gt.append(torch.from_numpy(batch[1]).to(device))

            
            #Triggers the computation of the Batch once it is filed up or the Last entry of the testdataset is reached
            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(testbatches) - 1:
                batch_in = torch.stack(batch_in)
                
                Model.reset_states()

                if batch_idx == len(testbatches)-1:
                    break

                counter = sinabs.SNNSynOpCounter(Model.spiking_model)
                    
                #adds channel dimension
                batch_in = torch.unsqueeze(batch_in, dim = 1)
                            
                # Adds a time dimension
                batch_in = torch.unsqueeze(batch_in, dim = 1)

                # stacks the input along the time dimesion to a toatal length of TIMESTEPSSNNSIM
                batch_in_new = batch_in
                for i in range(TIMESTEPSSNNSIM-1):
                    batch_in_new = torch.cat((batch_in_new,batch_in), dim = 1)
                
                batch_in = batch_in_new

                # Flattens time dimension into Batch dimension which is how the Sinab Library requires it for the forward pass
                batch_in = sinabs.layers.FlattenTime()(batch_in).to(torch.float32)
                
                    
                
                # create the ground truth tensor and normalize it
                batch_gt = torch.stack(batch_gt)
                batch_gt = torch.squeeze(batch_gt)
            
                firsttime = datetime.now()

                batch_out = Model(batch_in)
         

                # restore time dimension in output
                if len(batch_out.size()) == 4:
                    batch_out = batch_out.unsqueeze(dim = 1)

                
                SynopsList.append(counter.get_total_synops())

                batch_out = batch_out.unflatten(
                    0, (batch_size,batch_out.shape[0] // batch_size)
                    )
                
                
                batch_out = batch_out.squeeze()

                if len(batch_out.shape) == 1:
                    batch_out = batch_out.unsqueeze(dim = 0)
                if len(batch_out.shape) == 2:
                    batch_out = batch_out.unsqueeze(dim = 1)
                
                batch_out = torch.mean(batch_out, 1)

                

                if len(batch_gt.size()) == 1:
                    batch_gt = batch_gt.unsqueeze(dim = 0)

                if len(batch_out.size()) == 1:
                    batch_out = batch_out.unsqueeze(dim = 0)

                PredX = torch.argmax(batch_out[:,0:128],dim = 1)
                PredY = torch.argmax(batch_out[:,128:256],dim = 1)

                
                Model.reset_states()
                zero_dummy = torch.zeros((1,1))
                zero_dummy[0][0] = (datetime.now() - firsttime).microseconds
                batch_time_list.append(zero_dummy)

                batch_euclid_list.append(torch.sqrt((PredX-batch_gt[:,0])**2 + (PredY-batch_gt[:,1])**2))
                loss = 0

                
                with torch.no_grad():
                    total_loss += loss

                # delete torch variables to free up memory
                del(loss)
                del(batch_in)
                del(batch_gt)
                del(batch_out)
                batch_in = []
                batch_gt = []
    euclid = torch.cat(batch_euclid_list, dim = 0)
    timelist = torch.cat(batch_time_list, dim = 0)
    euclid_np = euclid.numpy()
    timelist_np = timelist.numpy()
    np.savetxt('euclid_' + str(runNum) + '.csv', euclid_np, delimiter = ',')
    np.savetxt('time_' + str(runNum) + '.csv', timelist_np, delimiter = ',')

    return (torch.mean(euclid), torch.std(euclid), torch.mean(torch.stack(SynopsList)), TIMESTEPSSNNSIM)
        



      

def test_DynapCNN(Snn, testbatches):
    """
    Testing function for DynapCNN Model. Takes a normal NN and turns ito a DynapCNN model, then tests it in simulation 
    or on the chip and returns the loss.

    Parameters:
        Snn:      torch.nn.Sequential;       the SNN we convert
        testbatches:   torch.tensor;         the teat data and ground truth
    """

    
    print(dir(Snn.spiking_model[6]))
    Snn.spiking_model[6].stride = (8,8)
    print(Snn)
    
    # Converts the SNN to a dynapCNN network, discretizes weights, thresholds
    DynapCNNModel = convert_to_DynapCNN(Snn)

    # Push model onto DynapCNN chip
    device = "dynapcnndevkit:0"
    DynapCNNModel.to(device)

    return(testOnChipFastDesync(DynapCNNModel, testbatches))   
    




def testOnChipFastDesync(Model, testbatches):
    """
    Function for testing a single Model on the chip, the Model must already be on the chip.

    parameters:
        Model:         DynapCNN model;   already located on the Chip
        testbatches:   torch.tensor;     the testdata and ground truth

    """
    time.sleep(1)
    _ = Model.samna_output_buffer.get_events() 
    del(_)
    # defines the samna device we use 
    device = 'dynapcnndevkit:0'
    # initializes various variables
    batch_in = []
    batch_gt = []
    batch_euclid_list = []

    enable_timestamps(Model.device)

    inputTimeList = []
    outputTimeList = []

    inputList = []
    gtList = []
    output_accumulate = []

    listLastTimestamp = []

    sequencenum = -1
    inputsequencenum = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testbatches):
            batch_in = []
            batch_gt = []
            batch_in.append(batch[0])
            batch_gt.append(batch[1])
            
            # optionally stops function early since testing on chip is slower
            #if batch_idx > 10:#len(testbatches)-1:
            #    break
            
            print(datetime.now().time())
            batch_in = torch.stack(batch_in)

            # adds channel and time dimension
            batch_in = torch.unsqueeze(batch_in, dim = 1)
            batch_in = torch.unsqueeze(batch_in, dim = 1)

            # Generates the repeated input for the model and reformats it
            batch_in = batch_in.expand(1,TIMESTEPSONCHIP,1, 64, 64) 
            batch_in_new = sinabs.layers.FlattenTime()(batch_in)


            # formatting of ground truth
            batch_gt = torch.stack(batch_gt)
            batch_gt = torch.squeeze(batch_gt)

            gtList.append(batch_gt)
            
            # Turns input frames into events that can be processed by the chip
            factory = ChipFactory(device)
            first_layer_idx = Model.chip_layers_ordering[0] 

            events_in = factory.raster_to_events(batch_in_new, layer=first_layer_idx, dt = 0.00001)

            inputList.append(events_in)
        len_inputlist = len(inputList)
        while(sequencenum < len_inputlist-1):
            
            Firsttime = datetime.now() 

            events_in = inputList[inputsequencenum]
            if(inputsequencenum < len_inputlist-1):
                inputsequencenum += 1

            print("start")
  
            inputTimeList.append(datetime.now())

            '''To use this a change has to made to the Sinabs library in sinabs/backend/dynapcnn/dynapcnn_network.forward
            if statement content should be (disable buffer flush and sleep): 
            
            #_ = self.samna_output_buffer.get_events()  # Flush buffer
            # NOTE: The code to start and stop time stamping is device specific
            reset_timestamps(self.device)
            enable_timestamps(self.device)
            # Send input
            self.samna_device.get_model().write(x)
            received_evts = []
            #time.sleep(0.1)
            while True:
                prev_length = len(received_evts)
                #time.sleep(0.1)
                received_evts.extend(self.samna_output_buffer.get_events())
                if prev_length == len(received_evts):
                    break
                break
            # Disable timestamp
            disable_timestamps(self.device)
            return received_evts

            '''
            events_out = Model(events_in)

            output_accumulate.extend(events_out)

            sequencecomplete = False
            lastneurontimestamp = 0
            index = 0
            #for index, event in enumerate(output_accumulate):
            while not(sequencecomplete) and index < len(output_accumulate):
                for event in output_accumulate:
                    if event.timestamp < lastneurontimestamp:#isinstance(event, builder.get_samna_module().event.NeuronValue):
                        events_out = output_accumulate[0:index]
                        
                        output_accumulate = output_accumulate[index:-1]
                        sequencecomplete = True
                        sequencenum += 1
                        break
                    else:
                        #print(events_out)
                        lastneurontimestamp = event.timestamp 
                        index +=1
                

            if not(sequencecomplete):
                time.sleep(0.005)

            if sequencecomplete:
                listLastTimestamp.append(events_out[-1].timestamp)
                
                batch_gt = gtList[sequencenum]

                #turn the outut back into a torch tensor 
                if len(events_out) != 0:
                    # shape is (chanel, width ,height)
                    batch_out = factory.events_to_raster(events_out,dt = 0.01,shape = (256,1,1))
                else:
                    batch_out = torch.zeros((256,1,1))
                
                batch_out = batch_out.float()

                batch_out = torch.squeeze(batch_out)
 
                # calculate the ratecode of the output
            
                if len(batch_out.size()) > 1:
                    batch_out = torch.sum(batch_out,dim = 0)

                batch_outX = torch.argmax(batch_out[0:128])
                batch_outY = torch.argmax(batch_out[128:256])

                batch_euclid_list.append(torch.sqrt((batch_outX-batch_gt[0])**2 + (batch_outY-batch_gt[1])**2))

                outputTimeList.append(datetime.now())    
                 
                print("current batch:")
                print(sequencenum)
                print("mean euclidean distance")
                print(torch.mean(torch.stack(batch_euclid_list,dim = 0)))
                print(datetime.now().time())

                # deleting torch objects to free up memory

                del(batch_in)
                del(batch_gt)
                del(batch_out)
                batch_in = []
                batch_gt = []
                
                if batch_idx == 1000:
                    break

    print(datetime.now() - Firsttime)
    print(listLastTimestamp)

    diffTimeList = []
    print(len(inputTimeList))
    print(len(outputTimeList))
    for index in range(0,len(outputTimeList)):
        diffTimeList.append((outputTimeList[index] - inputTimeList[index]).total_seconds())
    
    print(np.mean(diffTimeList))
    print((outputTimeList[-1] - inputTimeList[0]).total_seconds()/len(outputTimeList))

    np_output_chip = np.zeros((4,2))
    np_output_chip[0,0] = torch.mean(torch.stack(batch_euclid_list,dim = 0))
    np_output_chip[0,1] = torch.std(torch.stack(batch_euclid_list,dim = 0))
    np_output_chip[1,0] = np.mean(listLastTimestamp[1:-1])
    np_output_chip[1,1] = np.std(listLastTimestamp[1:-1])
    np_output_chip[2,0] = np.mean(diffTimeList)
    np_output_chip[2,1] = np.std(diffTimeList)
    np_output_chip[3,0] = (outputTimeList[-1] - inputTimeList[0]).total_seconds()/len(outputTimeList)
    print(np_output_chip)
    
    return np_output_chip

