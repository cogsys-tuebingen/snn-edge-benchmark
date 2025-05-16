#import os
import numpy as np
import typing as ty
import pandas as pd
from datetime import datetime


# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg, Loihi1HwCfg

from prepare_data import load_data

shape = (64,64,1)

np.set_printoptions(linewidth=np.inf)

data_folder = 'data/'

NUMBEROFTIMEFRAMES = 531


#RUN CONFIG IN SIM
run_config = Loihi1SimCfg()
#RUN CONFIG ON CHIP
#run_config = Loihi2HwCfg()


data, groundtruth = load_data(data_folder)
data, groundtruth = data[0:NUMBEROFTIMEFRAMES], groundtruth[0:NUMBEROFTIMEFRAMES]


class SpikeInput(AbstractProcess):
    """Reads image data from the MNIST dataset and converts it to spikes.
    The resulting spike rate is proportional to the pixel value."""

    def __init__(self,
                 vth: int,
                 num_images: ty.Optional[int] = 25,
                 num_steps_per_image: ty.Optional[int] = 128):
        super().__init__()
        
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the classifier
        self.label_out = OutPort(shape=(1,2))  # Ground truth labels to OutputProc
        self.num_images = Var(shape=(1,), init=num_images)
        self.num_steps_per_image = Var(shape=(1,), init=num_steps_per_image)
        self.input_img = Var(shape=shape)
        self.ground_truth_label = Var(shape=(1,2))
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)


from lava.lib.dl.netx import hdf5

net = hdf5.Network(net_config='slayer_network.net')



print(dir(net))
print(dir(net.in_layer))
print(dir(net.in_layer.neuron))
print(net.in_layer.neuron.u)
print(net.in_layer.neuron.v)
print(net.in_layer.neuron.vth)
print(net.in_layer.neuron.log)

print(net.layers)
#print(net.layers[1])
#print(dir(net.layers[1]))
#print(dir(net.layers[1].neuron))
#print(net.layers[1].neuron.u)

#print(net.layers[2].vars)
#print(net.layers[2].vars.member_names)

#print(net.layers[2])
#print(net.layers[2].synapse)
#print(dir(net.layers[2].synapse))
#print(net.layers[2].synapse.weights)
#print(dir(net.layers[1].synapse))
#print(net.layers[1].synapse.weight)


#net.in_layer.neuron.vth = 1

#net.layers[1].neuron.vth = 1
#net.layers[2].neuron.vth = 1

num_steps_per_image = 20


class OutputProcess(AbstractProcess):
    """Process to gather spikes from 10 output LIF neurons and interpret the
    highest spiking rate as the classifier output"""

    def __init__(self, **kwargs):
        super().__init__()
        shape = (1,1,256)
        n_img = kwargs.pop('num_images', 25)
        self.num_images = Var(shape=(1,), init=n_img)
        self.spikes_in = InPort(shape=shape)
        self.label_in = InPort(shape=(1,2))
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification
        self.num_steps_per_image = Var(shape=(1,), init=num_steps_per_image)
        self.pred_labels = Var(shape=(n_img,2))
        self.gt_labels = Var(shape=(n_img,2))


# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import Loihi2NeuroCore

from lava.utils.dataloader.mnist import MnistDataset
print(np.reshape(MnistDataset().images[1], (28,28)))
np.set_printoptions(linewidth=np.inf)


@implements(proc=SpikeInput, protocol=LoihiProtocol)
@requires(CPU)
#@requires(CPU)
class PySpikeInputModel(PyLoihiProcessModel):
    num_images: int = LavaPyType(int, int, precision=32)
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=1)
    label_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32,
                                      precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    input_img: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    ground_truth_label: int = LavaPyType(int, int, precision=32)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.mnist_dataset = MnistDataset()
        self.curr_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % self.num_steps_per_image == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        #print(self.time_step)
        img = np.expand_dims(data[self.curr_img_id],axis = 2)

        img = np.expand_dims(np.transpose(data[self.curr_img_id + 0]),axis = 2) 
        
        self.ground_truth_label = np.squeeze(groundtruth[self.curr_img_id + 0])
        self.input_img = img.astype(np.int32) #- 127
        self.v = np.zeros(self.v.shape)
        self.label_out.send(np.array([self.ground_truth_label]))
        self.curr_img_id += 1

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        self.v = self.v + self.input_img
        s_out = self.v > self.vth
        self.v[s_out] = 0  # reset voltage to 0 after a spike
        #print(np.squeeze(s_out))
        #print('a')
        self.spikes_out.send(s_out)
        #self.spikes_out.send(self.input_img)

'''
from lava.magma.core.resources import LMT # Embedded CPU
from lava.magma.core.model.c.type import LavaCType, LavaCDataType, COutPort
from lava.magma.core.model.c.model import CLoihiProcessModel
@implements(proc=SpikeInput, protocol=LoihiProtocol)
@requires(LMT)
class CSpikeGeneratorModel(CLoihiProcessModel):
    """Spike Generator process model in C."""

    num_images: Var = LavaCType(cls = int, d_type=LavaCDataType.INT32)
    spikes_out: COutPort = LavaPyType(cls=COutPort, d_type=LavaCDataType.INT32)
    label_out: COutPort = LavaPyType(Pcls=COutPort, d_type=LavaCDataType.INT32)
    num_steps_per_image: Var = LavaCType(cls = int, d_type=LavaCDataType.INT32)
    input_img: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    ground_truth_label: int = LavaPyType(int, int, precision=32)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)


    spike_prob: Var = LavaCType(cls=int, d_type=LavaCDataType.INT32)
    s_out: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
    
    @property
    def source_file_name(self):
        return "spike_generator.c"
'''

@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputProcessModel(PyLoihiProcessModel):
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    num_images: int = LavaPyType(int, int, precision=32)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    pred_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    gt_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
        
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.current_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % self.num_steps_per_image == 0 and \
                self.time_step > 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        firsttime = datetime.now()
        gt_label = np.squeeze(self.label_in.recv())
        pred_label = (np.argmax(np.squeeze(self.spikes_accum[0,0,0:128])),np.argmax(np.squeeze(self.spikes_accum[0,0,128:256])))
        #print(gt_label)
        #print(pred_label)
        self.gt_labels[self.current_img_id] = gt_label
        self.pred_labels[self.current_img_id] = pred_label
        self.current_img_id += 1
        self.spikes_accum = np.zeros_like(self.spikes_accum)
        print('time for post managment')
        print(datetime.now() - firsttime)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        print('input_spikes')
        #print(spk_in)
        spk_in = np.squeeze(spk_in)
        print(spk_in)
        self.spikes_accum = self.spikes_accum + spk_in
        #print(self.spikes_accum.shape)



num_images = 531


spike_input = SpikeInput(vth=0.5,
                        num_steps_per_image=num_steps_per_image,
                        num_images=num_images)
output_proc = OutputProcess(num_images=num_images)



from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol





from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from lava.proc.conv.process import Conv





neuron_0_u = net.layers[0].neuron.u
print(neuron_0_u.shape)
neuron_0_du = net.layers[0].neuron.du
print(neuron_0_du)
neuron_0_dv = net.layers[0].neuron.dv
print(neuron_0_dv)
neuron_0_bias_mant = net.layers[0].neuron.bias_mant 
print(neuron_0_bias_mant)
neuron_0_bias_exp = net.layers[0].neuron.bias_exp
print(neuron_0_bias_exp)

lif_0 = LIF(shape = neuron_0_u.shape, 
            du = neuron_0_du.get(), 
            dv = neuron_0_dv.get(),
              bias_mant = neuron_0_bias_mant.get(),
                bias_exp = neuron_0_bias_exp.get())
lif_0_alternative = LIF(shape = (64,64,1), bias_mant = 100000, bias_exp = 6, vth = 1)

synapse_1_weight = net.layers[1].synapse.weight
#print(synapse_1_weight)
synapse_1_weight_exp = net.layers[1].synapse.weight_exp
print(synapse_1_weight_exp)
synapse_1_input_shape = net.layers[1].synapse.input_shape
print(synapse_1_input_shape)
synapse_1_padding = net.layers[1].synapse.padding
print(synapse_1_padding)
print(synapse_1_padding.get())
synapse_1_stride = net.layers[1].synapse.stride
print(synapse_1_stride)

conv_1 = Conv(weight = synapse_1_weight.get(), weight_exp = synapse_1_weight_exp.get(), input_shape = synapse_1_input_shape, padding = synapse_1_padding.get(), stride = synapse_1_stride.get())
#conv_1 = Conv(weight = synapse_1_weight.get(), input_shape = synapse_1_input_shape, stride = 2)


neuron_1_u = net.layers[1].neuron.u
print(neuron_1_u.shape)
neuron_1_du = net.layers[1].neuron.du
print(neuron_1_du)
neuron_1_dv = net.layers[1].neuron.dv
print(neuron_1_dv)
neuron_1_bias_mant = net.layers[1].neuron.bias_mant 
print(neuron_1_bias_mant)
neuron_1_bias_exp = net.layers[1].neuron.bias_exp
print(neuron_1_bias_exp)

#lif_1 = LIF(shape = neuron_1_u.shape,
#             du =  neuron_1_du.get(),
#             dv = neuron_1_dv.get(),
#             bias_mant = neuron_1_bias_mant.get(), 
#             bias_exp = neuron_1_bias_exp.get())
lif_1 = LIF(shape = neuron_1_u.shape,
             du =  neuron_1_du.get(),
             dv = neuron_1_dv.get(),
             bias_mant = 100000, bias_exp = 6)
#lif_1 = LIF(shape = neuron_1_u.shape)


synapse_2_weight = net.layers[2].synapse.weight
#print(synapse_2_weight)
synapse_2_weight_exp = net.layers[2].synapse.weight_exp
print(synapse_2_weight_exp)
synapse_2_input_shape = net.layers[2].synapse.input_shape
print(synapse_2_input_shape)
synapse_2_padding = net.layers[2].synapse.padding
print(synapse_2_padding)
print(synapse_2_padding.get())
synapse_2_stride = net.layers[2].synapse.stride
print(synapse_2_stride)

conv_2 = Conv(weight = synapse_2_weight.get(), weight_exp = synapse_2_weight_exp.get(), input_shape = synapse_2_input_shape, padding = synapse_2_padding.get(), stride = synapse_2_stride.get())

neuron_2_u = net.layers[2].neuron.u
print(neuron_2_u.shape)
neuron_2_du = net.layers[2].neuron.du
print(neuron_2_du)
neuron_2_dv = net.layers[2].neuron.dv
print(neuron_2_dv)
neuron_2_bias_mant = net.layers[2].neuron.bias_mant 
print(neuron_2_bias_mant)
neuron_2_bias_exp = net.layers[2].neuron.bias_exp
print(neuron_2_bias_exp)

#lif_2 = LIF(shape = neuron_2_u.shape, 
#            du = neuron_2_du.get(), 
#            dv = neuron_2_dv.get(),
 #             bias_mant = neuron_2_bias_mant.get(), bias_exp = neuron_2_bias_exp.get())
lif_2 = LIF(shape = neuron_2_u.shape, 
            du = neuron_2_du.get(), 
            dv = neuron_2_dv.get(),
             bias_mant = 100000, bias_exp = 6)


synapse_3_weight = net.layers[3].synapse.weight
#print(synapse_3_weight)
synapse_3_weight_exp = net.layers[3].synapse.weight_exp
print(synapse_3_weight_exp)
synapse_3_input_shape = net.layers[3].synapse.input_shape
print(synapse_3_input_shape)
synapse_3_padding = net.layers[3].synapse.padding
print(synapse_3_padding)
print(synapse_3_padding.get())
synapse_3_stride = net.layers[3].synapse.stride
print(synapse_3_stride)

conv_3 = Conv(weight = synapse_3_weight.get(), weight_exp = synapse_3_weight_exp.get(), input_shape = synapse_3_input_shape, padding = synapse_3_padding.get(), stride = 16)#synapse_3_stride.get())

neuron_3_u = net.layers[3].neuron.u
print(neuron_3_u.shape)
neuron_3_du = net.layers[3].neuron.du
print(neuron_3_du)
neuron_3_dv = net.layers[3].neuron.dv
print(neuron_3_dv)
neuron_3_bias_mant = net.layers[3].neuron.bias_mant 
print(neuron_3_bias_mant)
neuron_3_bias_exp = net.layers[3].neuron.bias_exp
print(neuron_3_bias_exp)

#lif_3 = LIF(shape = neuron_3_u.shape, 
#            du = neuron_3_du.get(), 
#            dv = neuron_3_dv.get(), 
#            bias_mant = neuron_3_bias_mant.get(), bias_exp = neuron_3_bias_exp.get())
lif_3 = LIF(shape = neuron_3_u.shape, 
            du = neuron_3_du.get(), 
            dv = neuron_3_dv.get(), 
            bias_mant = 100000, bias_exp = 6)

synapse_4_weight = net.layers[4].synapse.weight
#print(synapse_3_weight)
synapse_4_weight_exp = net.layers[4].synapse.weight_exp
print(synapse_4_weight_exp)
synapse_4_input_shape = net.layers[4].synapse.input_shape
print(synapse_4_input_shape)
synapse_4_padding = net.layers[4].synapse.padding
print(synapse_4_padding)
print(synapse_4_padding.get())
synapse_4_stride = net.layers[4].synapse.stride
print(synapse_4_stride)

conv_4 = Conv(weight = synapse_4_weight.get(), weight_exp = synapse_4_weight_exp.get(), input_shape = synapse_4_input_shape, padding = synapse_4_padding.get(), stride = 1)#synapse_3_stride.get())


neuron_4_u = net.layers[4].neuron.u
print(neuron_4_u.shape)
neuron_4_du = net.layers[4].neuron.du
print(neuron_4_du)
neuron_4_dv = net.layers[4].neuron.dv
print(neuron_4_dv)
neuron_4_bias_mant = net.layers[4].neuron.bias_mant 
print(neuron_4_bias_mant)
neuron_4_bias_exp = net.layers[4].neuron.bias_exp
print(neuron_4_bias_exp)

#lif_4 = LIF(shape = neuron_4_u.shape, 
#            du = neuron_3_du.get(), 
#            dv = neuron_3_dv.get(), 
#            bias_mant = neuron_4_bias_mant.get(), bias_exp = neuron_4_bias_exp.get())
lif_4 = LIF(shape = neuron_4_u.shape, 
            du = neuron_3_du.get(), 
            dv = neuron_3_dv.get(), 
            bias_mant = 100000, bias_exp = 6)


input_layer = Conv(weight=np.ones((1,1,1,1)), input_shape = (64,64,1))   
print(input_layer.a_out.shape)

#CONNECT MANUAL LAYERS
print(input_layer.a_out.shape)
print(lif_0.a_in.shape)


input_layer.a_out.connect(lif_0.a_in)
#alt
#input_layer.a_out.connect(lif_0_alternative.a_in)
#lif_0_alternative.s_out.connect(conv_1.s_in)
#endalt
lif_0.s_out.connect(conv_1.s_in)
conv_1.a_out.connect(lif_1.a_in)
lif_1.s_out.connect(conv_2.s_in)
conv_2.a_out.connect(lif_2.a_in)
lif_2.s_out.connect(conv_3.s_in)
conv_3.a_out.connect(lif_3.a_in)
lif_3.s_out.connect(conv_4.s_in)
conv_4.a_out.connect(lif_4.a_in)

#connect to input and output
from lava.proc import embedded_io as eio

inp_adapter = eio.spike.PyToN3ConvAdapter(shape=(64,64,1))
out_adapter = eio.spike.NxToPyAdapter(shape=(1, 1, 256))



#spike_input.spikes_out.connect(lif_0.a_in)
spike_input.spikes_out.connect(inp_adapter.inp)
inp_adapter.out.connect(input_layer.s_in)
lif_4.s_out.connect(out_adapter.inp)
#spike_input.label_out.connect(output_proc.label_in)
out_adapter.out.connect(output_proc.spikes_in)



from lava.magma.core.run_conditions import RunSteps

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.compiler.compiler import Compiler


from lava.proc.spiker.process import Spiker

print(lif_0.a_in.shape)


img = np.expand_dims(np.transpose(data[0]),axis = 2) 
spiker_input = Spiker(shape = (64, 64,1), payload =1, period = 1)

print(spiker_input.payload.shape)
print(spiker_input.s_out.shape)

#spiker_input.spikes_out.connect(lif_0.a_in)
#spiker_input.s_out.connect(inp_adapter.inp)
#spiker_input.s_out.connect(output_proc.spikes_in)
#inp_adapter.out.connect(input_layer.s_in)
#lif_3.s_out.connect(out_adapter.inp)
#spike_input.label_out.connect(output_proc.label_in)
#out_adapter.out.connect(output_proc.spikes_in)

#spiker_input.s_out.connect(lif_0_alternative.a_in)

# create a compiler
#compiler = Compiler()

#test = compiler.compile(spiker_input, run_cfg = Loihi2HwCfg())

#executable = compiler.compile(lif_0, run_cfg = Loihi2HwCfg())

from lava.magma.runtime.runtime import Runtime
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.process.message_interface_enum import ActorType

from lava.utils.profiler import Profiler



profiler = Profiler.init(run_config)
profiler.execution_time_probe(num_steps=num_steps_per_image)

# create and initialize a runtime
mp = ActorType.MultiProcessing
#runtime = Runtime(exe=executable,
                  #message_infrastructure_type=mp)
#runtime.initialize()

# Loop over all images
for img_id in range(num_images):
    print(f"\rCurrent image: {img_id+1}", end="")

  
    
    firsttime = datetime.now()
    
    

    lif_3.run(
        condition=RunSteps(num_steps=num_steps_per_image),
        run_cfg=run_config)#Loihi1SimCfg(select_tag='fixed_pt'))
    #lif_0.stop()
    
    #runtime.start(run_condition=RunSteps(num_steps=num_steps_per_image))
    #runtime.stop()
    print(datetime.now() - firsttime)


    #if img_id == 0:
        #break


   

# Gather ground truth and predictions before stopping exec
ground_truth = output_proc.gt_labels.get().astype(np.int32)
predictions = output_proc.pred_labels.get().astype(np.int32)

print(output_proc.spikes_accum.get().astype(np.int32))






# Stop the execution
lif_3.stop()

print('execution time')
print(profiler.execution_time)
print('mean execution time')
print(np.mean(profiler.execution_time))
print('spiking_time')
print(profiler.spiking_time)
print('mean spiking_time')
print(np.mean(profiler.spiking_time))
print('management_time')
print(profiler.management_time)
print('mean management_time')
print(np.mean(profiler.management_time))
print('host_time')
print(profiler.host_time)
print('mean host_time')
print(np.mean(profiler.host_time))

accuracy = np.sum(ground_truth==predictions)/ground_truth.size * 100

print(np.sqrt(np.sum(np.square(ground_truth - predictions), axis = 1)))

euclid = np.mean(np.sqrt(np.sum(np.square(ground_truth - predictions), axis = 1)))

print(f"\nGround truth: {ground_truth}\n"
      f"Predictions : {predictions}\n"
      f"Accuracy    : {accuracy}"
      f"Mean euclidean distance : {euclid}")

print(groundtruth)
