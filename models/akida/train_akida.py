import sys
sys.path.append('../../data_processing/')

import numpy as np
import os
import tensorflow as tf
import pandas as pd
from datetime import datetime
import random
import time
import cnn2snn
from prepare_data import load_data



SNNBATCHSIZE = 100

NUMBEROFTIMEFRAMES = 7569

NUMBEROFTIMEFRAMES_HL = 2123

data_folder_projected = 'dataset/projected/'
data_folder_handlabeled = 'dataset/hand_labeled/'


def test(Model, testbatches, SNN = False):

    batch_euclid_list = []

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

                print(batch_in.shape)

                batch_out = Model.predict(batch_in.numpy().astype('uint8'))

                print(datetime.now() - firsttime)
     
                batch_out = tf.convert_to_tensor(batch_out)
                batch_out = tf.squeeze(batch_out)

            else:
                firsttime = datetime.now()

                print(batch_in.shape)

                batch_out = Model(batch_in)

                print(datetime.now() - firsttime)
                
            if len(batch_gt.shape) == 1:
                batch_gt_np = np.expand_dims(batch_gt.numpy(),axis = 0)
                batch_gt = tf.convert_to_tensor(batch_gt_np)

            if len(batch_out.shape) == 1:
                batch_out_np = np.expand_dims(batch_out.numpy(),axis = 0)
                batch_out = tf.convert_to_tensor(batch_out_np)

            PredX = tf.math.argmax(batch_out[:,0:128],axis = 1)
            PredY = tf.math.argmax(batch_out[:,128:256],axis = 1)
            
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
    print("mean euclidian distance")
    print(tf.math.reduce_mean(euclid))
    total_loss = tf.math.reduce_mean(euclid)
    print("std of euclidian distance")
    print(tf.math.reduce_std(euclid))

    return total_loss/(len(testbatches)/batch_size)
    
    

def trainEpochs(Model, Trainbatches, Testbatches, loss_function, trained_network_file = "dummy"):

    nr_epochs = 100
    
    batch_size = 1000
    
    start_time = time.time()

    increasestreak = 0
    lasttestloss = tf.convert_to_tensor(np.array(100000.0))

    for epoch in range(nr_epochs):
        
        random.shuffle(Trainbatches)
 
        earlystoppingcrit = 10

        total_loss = 0
        batch_in = []
        batch_gt = []
        
        testloss = test(Model,Testbatches)

        if lasttestloss > testloss:
            lasttestloss = testloss
            increasestreak = 0
            Model.save(trained_network_file)
            model_akida = cnn2snn.convert(Model)
            model_akida.save(trained_network_file + '.fbz')
        else:
            increasestreak += 1
            print("best so far:")
            print(lasttestloss)
            
        if increasestreak >= earlystoppingcrit:
            print("EarlyStopping")
            break
        
        for batch_idx, batch in enumerate(Trainbatches):
            batch_in.append(batch[0])
            batch_gt.append(batch[1])

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(Trainbatches) - 1:

                batch_in = tf.stack(batch_in)
                batch_in = tf.expand_dims(batch_in, axis = 3)

                batch_gt = tf.stack(batch_gt)
                batch_gt = tf.squeeze(batch_gt)

                batch_out = Model(batch_in)
                batch_out = tf.squeeze(batch_out)
                
                batch_gtX = tf.round(batch_gt[:,0]*1)
                batch_gtY = tf.round(batch_gt[:,1]*1)

                target = tf.Variable(tf.zeros(batch_out.shape))
                for i in range(batch_out.shape[0]):
                    if batch_gtX[i].numpy() > -1:
                        gtX_value = batch_gtX[i].numpy()
                        gtY_value = batch_gtY[i].numpy()
                        
                        if gtX_value > -1:
                            # Prepare the indices and values for batch_gtX
                            indices_X = [
                                [i, int(gtX_value)], 
                                [i, int(gtX_value - 1)], 
                                [i, int(gtX_value + 1)]
                            ]
                            values_X = [1.0, 0.5, 0.5]
                            
                            # Prepare the indices and values for batch_gtY
                            indices_Y = [
                                [i, int(128 + gtY_value)], 
                                [i, int(128 + gtY_value - 1)], 
                                [i, int(128 + gtY_value + 1)]
                            ]
                            values_Y = [1.0, 0.5, 0.5]

                            # Combine the indices and values for both batch_gtX and batch_gtY
                            indices = indices_X + indices_Y
                            values = values_X + values_Y
                            
                            # Convert to tensors
                            indices_tensor = tf.constant(indices)
                            values_tensor = tf.constant(values)

                            # Reassign the updated tensor back to target
                            target = tf.tensor_scatter_nd_update(target, indices_tensor, values_tensor)

                    target = tf.convert_to_tensor(target)

                    loss = loss_function(batch_out,target)
                          
                total_loss = loss
                
                Model.fit(batch_in, target)

                del(batch_in)
                del(batch_gt)
                del(batch_out)
                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print(time_left)
        print(total_loss.numpy())
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, 
            total_loss.numpy(), 
            time_left))
        
def trainModel(trained_network_file):
    with tf.device('/CPU:0'):

        device = tf.device('/CPU:0')

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(4, 5, strides = (2,2), activation = 'linear', input_shape = (64,64,1), padding='valid'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(padding='valid'),
            tf.keras.layers.Conv2D(4, 3, strides = (2,2), activation = 'linear', input_shape = (15,15,4), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation = 'linear'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(256, activation = 'linear')
        ])

        model.summary()

        # Projected data
        projected_frames, projected_positions = load_data(data_folder_projected)

        # Handlabeled data
        handlabeled_frames, handlabeled_positions = load_data(data_folder_handlabeled)

        basedata, baseTruth = projected_frames, projected_positions
        data_hl, gt_hl = handlabeled_frames, handlabeled_positions

        train_hl = data_hl[round((len(data_hl)*0.5)):round((len(data_hl)*1))]
        train_hl_gt = gt_hl[round((len(data_hl)*0.5)):round((len(data_hl)*1))]

        test_hl = data_hl[round((len(data_hl)*0)):round((len(data_hl)*0.25))]
        test_hl_gt = gt_hl[round((len(data_hl)*0)):round((len(data_hl)*0.25))]

        basedata = tf.concat([basedata, train_hl], axis = 0)
        baseTruth = tf.concat([baseTruth, train_hl_gt], axis = 0)

        Trainbatches = [batch for batch in zip(basedata, baseTruth)]
        Testbatches = [batch for batch in zip(test_hl, test_hl_gt)]

        loss_function = tf.keras.losses.MeanSquaredError()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model = cnn2snn.quantize(model, weight_quantization = 4, activ_quantization = 4, input_weight_quantization = 2)
        model.compile(loss = tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            decay = 0.0001 
            ))

        trainEpochs(model, Trainbatches, Testbatches, loss_function, trained_network_file = trained_network_file)
