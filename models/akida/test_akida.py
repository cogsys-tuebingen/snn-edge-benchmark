import numpy as np
import tensorflow as tf
import pandas as pd
import torch
from datetime import datetime
from prepare_data import load_data

SNNBATCHSIZE = 1

NUMBEROFTIMEFRAMES_HL = 2123
NUMBEROFTIMEFRAMES_PROJ = 7569

data_folder_handlabeled = 'dataset/hand_labeled/'


def test(Model, testbatches, SNN = False, runNum = 666):

    batch_euclid_list = []
    batch_time_list = []
    batch_time_list_chip = []
    
    with torch.no_grad():

        total_loss = 0

        batch_size = SNNBATCHSIZE
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(testbatches):
       
            batch_in.append(batch[0])
   
            batch_gt.append(batch[1])

   
            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(testbatches) - 1:

                batch_in = tf.stack(batch_in)
                batch_in = tf.expand_dims(batch_in, axis = 3)

                batch_gt = tf.stack(batch_gt)
                batch_gt = tf.squeeze(batch_gt)
            
                if SNN:
                    firsttime = datetime.now()

                    batch_out = Model.predict(batch_in.numpy().astype('uint8'))

                    zero_dummy = torch.zeros((1,1))
                    zero_dummy[0][0] = (datetime.now() - firsttime).microseconds

                    batch_time_list_chip.append(zero_dummy)

                    batch_out = tf.convert_to_tensor(batch_out)
                    batch_out = tf.squeeze(batch_out)

                else:
                    firsttime = datetime.now()
                    batch_out = Model(batch_in)

                if len(batch_gt.shape) == 1:
                    batch_gt_np = np.expand_dims(batch_gt.numpy(),axis = 0)
                    batch_gt = tf.convert_to_tensor(batch_gt_np)

                if len(batch_out.shape) == 1:
                    batch_out_np = np.expand_dims(batch_out.numpy(),axis = 0)
                    batch_out = tf.convert_to_tensor(batch_out_np)

                PredX = tf.math.argmax(batch_out[:,0:64],axis = 1)
                PredY = tf.math.argmax(batch_out[:,128:192],axis = 1)

                zero_dummy = torch.zeros((1,1))
                zero_dummy[0][0] = (datetime.now() - firsttime).microseconds

                batch_time_list.append(zero_dummy)
     
                batch_euclid_list.append(tf.math.sqrt((tf.cast(PredX, tf.float64)-batch_gt[:,0])**2 + (tf.cast(PredY, tf.float64)-batch_gt[:,1])**2))
                loss = 0


                total_loss += loss

                del(loss)
                del(batch_in)
                del(batch_gt)
                del(batch_out)
                batch_in = []
                batch_gt = []

    euclid = tf.concat(batch_euclid_list,axis = 0)
    print("euclidian distance")
    timelist = torch.cat(batch_time_list, dim = 0)
    timelist_chip = torch.cat(batch_time_list_chip, dim = 0)
    euclid_np = euclid.numpy()
    timelist_np = timelist.numpy()
    timelist_chip_np = timelist_chip.numpy()
    np.savetxt('euclid_' + str(runNum) + '.csv', euclid_np, delimiter = ',')
    np.savetxt('time_' + str(runNum) + '.csv', timelist_np, delimiter = ',')
    np.savetxt('time_chip_' + str(runNum) + '.csv', timelist_chip_np, delimiter = ',')
    print("mean euclidian distance")
    print(tf.math.reduce_mean(euclid))
    total_loss = tf.math.reduce_mean(euclid)
    print("std of euclidian distance")
    print(tf.math.reduce_std(euclid))

    return total_loss/(len(testbatches)/batch_size)
    


def testModel(Model, runNum):

    handlabeled_frames, handlabeled_positions = load_data(data_folder_handlabeled)
    handlabeled_frames = handlabeled_frames[0:NUMBEROFTIMEFRAMES_PROJ]
    handlabeled_positions = handlabeled_positions[0:NUMBEROFTIMEFRAMES_PROJ]

    test_hl = handlabeled_frames[round((len(handlabeled_frames)*0.25)):round((len(handlabeled_frames)*0.5))]
    test_hl_gt = handlabeled_positions[round((len(handlabeled_frames)*0.25)):round((len(handlabeled_frames)*0.5))]

    Testbatches = [batch for batch in zip(test_hl, test_hl_gt)]
    
    loss_function = tf.keras.losses.MeanSquaredError()
    test(Model, Testbatches, SNN=True, runNum=runNum)
