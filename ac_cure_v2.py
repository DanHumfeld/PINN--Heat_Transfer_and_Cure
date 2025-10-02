#################################################################
# Code      TGML_Heat Transfer
# Version   5.0
# Date      2021-12-27
# Author    Navid Zobeiry, Dan Humfeld, navidz@uw.edu
# Note      This code solves the heat equation with convective BC
#           Includes cure kinetics of *non-exotherming* resin
#           Python V3.6.8
#           Tensorflow V2.10
#
#################################################################
# Importing Libraries
#################################################################

# # Enable this section to hide all warnings and errors
# import os
# import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# import sys
# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')
# import absl.logging

import math
import random
import numpy as np
from numpy import savetxt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from time import time
start_time = time()

#################################################################
# Inputs 
#################################################################   
# File names
output_model = 'model_integrated.h5'
prediction_results = 'model_integrated.csv'
loss_history = 'loss_integrated.csv'

# Optimizer for training: SGD, RSMProp, Adagrad, Adam, Adamax...
learning_rate= 0.0005
my_optimizer = optimizers.Adam(learning_rate)
initializer = 'glorot_uniform'

# Mode: Train new = 0, continue training = 1, predict = 2
train_mode = 0

# Epoch and batch size control
epochs = 1200000 
initial_training_epochs = 1000         #20000
residual_test_epochs = 1000             #2000
batch = 1000      #1000
residual_test_point_count = 100

# Model hyper-parameters
nodes_per_layer = 32

# Normalization method: scaled by gradient with momentum
# Define the number of loss terms that will be used and the momentum (0-1) of the weighting of the loss terms
loss_terms = 7
loss_weighting_momentum = 0.9

# Mask method: generic, specific, or none 
#mask_method = 'generic'            # Any loss could be masked, but this is slow (comparison hasn't been done yet)                      [benchmark = 52-54s]
mask_method = 'specific'            # The masks and supporting structures are written only for the losses needing masks                 [benchmark = 20-22s]
#mask_method = 'none'               # Retains loss and gradients where they should be excluded (doc for the tooling material). May be appropriate to use at the beginning of model training. [benchmark = 11-12s]

#Save option 1: only top, mid, bot,  2:all virtual nodes
if_save = 2
plot_loss = False
time_reporting = False

#################################################################
# Heat Problem
#################################################################
T0 = 120 #120             # C   
tdot = 5/60          # C/s 
T_hold = 180 #180         # C 
t_hold = 36000 #36000       # S
h1 = 100             # W/m2K
h2 = 100 #50              # W/m2K

#################################################################
# Time Array
#################################################################
T_max = 240 #320 #T_hold
T_min = T0
t_min = 0
t_max = t_min + (T_hold-T0)/tdot + t_hold
time_kink = (T_hold-T0)/tdot

#################################################################
# Materials
#################################################################
L1 = 0.02 #0.2          # m
L2 = 0.01 #0.1          # m
L = L1 + L2         # m
transition_x = (L1)/(L1+L2)
transition_x_offset = 0.0001

#Steel
#k_mat = 51.9          # W/mK 
#rho_mat = 7859        # kg/m3
#cp_mat = 465          # J/(kg K)
#a_mat = k_mat / rho_mat / cp_mat      # m2/s

#Composite
k_mat1 = 0.47            # W/mK           <-- Should that be 0.148?, or 0.278? # https://www.wichita.edu/industry_and_defense/NIAR/Research/hexcel-8552/Additional-Documents-2.pdf
rho_mat1 = 1573          # kg/m3
cp_mat1 = 967            # J/(kg K)
a_mat1 = k_mat1 / rho_mat1 / cp_mat1      # m2/s
resin_volume_fraction = 0.45    # unitless
resin_density = 1301     # kg/m3
resin_heat_of_reaction = 6e5 * 20   # J/kg          <-- Multiplied by 20 because the response I was getting was underwhelming 
b_mat1 = resin_volume_fraction * resin_density * resin_heat_of_reaction / (rho_mat1 * cp_mat1)  # K 

# Artificially thermal insulation-enhanced cured composite
k_mat2 = 0.47/6        # W/mK    To make it insulating, divide by 6 
rho_mat2 = 1573          # kg/m3
cp_mat2 = 967            # J/(kg K)
a_mat2 = k_mat2 / rho_mat2 / cp_mat2     # m2/s      k/(rho cp)
b_mat2 = 0               # K

#Composite cure kinetics
doc_0_mat1 = 0.01                
doc_0_mat2 = 0.999                
doc_max = 1
doc_min = 0

AA=152800           #1/s
AA_norm = AA * (t_max - t_min)  # 1/1   About 9e8
Ea=66500            #J/mol              Ea/RT0 = 66500/(8.314*273.15) = 66500/2270 = 29.2. exp(-29.2) = 2e-13 = really rather zero. 
                    #                   Ea/RThold = 66500/(8.314*523.15) = 15.3. exp(-15.3) = 2.3e-7, still very small, then multiply by 1.5e5*6000 to get 180. It's non-zero and I guess it takes many seconds to evolve
mm=0.8129
nn=2.736
R = 8.3141          #J/mol.K

def CK2ln(T, x):          # This function takes in normalized T^ and x. 
    # This is meant to solve the issue of taking the log of x_dot when x_dot = 0 because x = 1, but it may eliminate the ability to take the gradient, so it's suppressed. Instead, fully cured is defined as 0.999 in the material.
    #x = np.array([min(k.get_value(x_item[0]), 0.999) for x_item in x])
    #x = x.reshape(len(x),1)
    #x = tf.convert_to_tensor(x)

    t1 = T * (T_max - T_min) + T_min + 273.15
    t2 = k.abs(x)
    t3 = k.abs(1-k.abs(x))     # This solves the nan issue that would occur if t3 were allowed to be negative
                                # This makes the formulas all invalid, but once the code works this won't be used so it's okay
    lnx_dot = tf.math.log(AA_norm)+(-Ea/(R*t1))+mm*tf.math.log(t2)+nn*tf.math.log(t3)
    x_dot = AA_norm*k.exp(-Ea/(R*t1))*k.pow(t2, mm)*k.pow(t3, nn)
    return lnx_dot
    #doc_dot = AA_norm*k.exp(-Ea/(R*t1))*k.pow(t2, mm)*k.pow(t3, nn)
    #return doc_dot

def a_lookup(xs):
    a = [a_mat1 if x < transition_x else a_mat2 for x in xs]
    return a

def b_lookup(xs):
    b = [b_mat1 if x < transition_x else b_mat2 for x in xs]
    return b

def k_lookup(xs):
    k = [k_mat1 if x < transition_x else k_mat2 for x in xs]
    return k

def doc_0_lookup(xs):
    doc_0 = [doc_0_mat1 if x < transition_x else doc_0_mat2 for x in xs]
    return doc_0

#################################################################
# Flatten method
#################################################################
def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

#################################################################
# Building or Load Model
#################################################################
if (train_mode == 0):
    input1 = keras.layers.Input(shape=(1,))         # x
    input2 = keras.layers.Input(shape=(1,))         # t

    layer1 = keras.layers.concatenate([input1, input2])
    layer2 = keras.layers.Dense(nodes_per_layer, activation='relu', kernel_initializer=initializer, bias_initializer=initializer)(layer1)   # This allows for a discontinuity in the slope of the temperature. This is important when using 2 materials.
    layer3 = keras.layers.Dense(nodes_per_layer, activation='elu', kernel_initializer=initializer, bias_initializer=initializer)(layer2)
    layer4 = keras.layers.Dense(nodes_per_layer, activation='elu', kernel_initializer=initializer, bias_initializer=initializer)(layer3)
    layer5 = keras.layers.Dense(nodes_per_layer, activation='elu', kernel_initializer=initializer, bias_initializer=initializer)(layer4)
    output = keras.layers.Dense(1, activation = 'softplus', kernel_initializer=initializer, bias_initializer=initializer)(layer5)           # Temperature

    layer11 = keras.layers.concatenate([input1, input2])
    layer21 = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer11)
    layer31 = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer21)
    layer41 = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer31)
    layer51 = keras.layers.Dense(nodes_per_layer, activation='tanh', kernel_initializer=initializer, bias_initializer=initializer)(layer41)
    output1 = keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer=initializer)(layer41)            # Degree of cure

    model = keras.models.Model([input1, input2], [output, output1])
    model.compile(loss='mse', optimizer=my_optimizer)
    
else:
    model = keras.models.load_model(output_model) 

#################################################################
# Main Code
#################################################################
if (train_mode < 2):
    # Set up loss weightings factors
    loss_weightings = np.ones(loss_terms)
    loss_weightings_previous = loss_weightings

    #Create Graph
    xdata = []
    ydata = []
    timedata = []
    if plot_loss:
        thatplot = plt.figure()
        thatplot.show()
        thatplot.patch.set_facecolor((0.1,0.1,0.1))
        axes = plt.gca()
        axes.set_xlim(0, 10)
        axes.set_ylim(0, +1)
        axes.set_facecolor((0.1,0.1,0.1))
        axes.spines['bottom'].set_color((0.9,0.9,0.9))
        axes.spines['top'].set_color((0.9,0.9,0.9))
        axes.spines['left'].set_color((0.9,0.9,0.9))
        axes.spines['right'].set_color((0.9,0.9,0.9))
        axes.xaxis.label.set_color((0.9,0.9,0.9))
        axes.yaxis.label.set_color((0.9,0.9,0.9))
        axes.tick_params(axis='x', colors=(0.9,0.9,0.9))
        axes.tick_params(axis='y', colors=(0.9,0.9,0.9))
        line, = axes.plot(xdata, ydata, 'r-') 

    min_loss = 100
    # This used to be inside of the loop, but with RRAR you only use a random set of points at the beginning
    
    # Create tensors to feed to TF
    # Time and Temperature are scaled from 0 - 1  
    x_arr = np.random.uniform(0, 1, batch) #*L
    t_arr = np.random.uniform(0, 1, batch) #*time_tot 
    last_time = time()
    for i in range(0, epochs):
        # When appropriate: add to batch
        if (i % residual_test_epochs == 0) and (i > initial_training_epochs):
            x_arr = np.append(x_arr, np.random.uniform(0, 1, residual_test_point_count))
            t_arr = np.append(t_arr, np.random.uniform(0, 1, residual_test_point_count))

        # Update all feeds at the beginning, when the batch has been increased, and when the batch has been decreased
        #if ((i==0) or ((i % residual_test_epochs == 0) and (i > initial_training_epochs)) or ((i % residual_test_epochs == 1) and (i > initial_training_epochs))):
        if ((i==0) or ((i % residual_test_epochs in [0,1]) and (i > initial_training_epochs))):
            x_feed = np.column_stack((x_arr)) 
            x_feed = tf.Variable(x_feed.reshape(len(x_feed[0]),1), trainable=True, dtype=tf.float32)

            t_feed = np.column_stack((t_arr)) 
            t_feed = tf.Variable(t_feed.reshape(len(t_feed[0]),1), trainable=True, dtype=tf.float32)

            zero_feed = np.column_stack(np.zeros(len(t_arr)))
            zero_feed = tf.Variable(zero_feed.reshape(len(zero_feed[0]),1), trainable=True, dtype=tf.float32)

            one_feed = np.column_stack(np.ones(len(t_arr)))
            one_feed = tf.Variable(one_feed.reshape(len(one_feed[0]),1), trainable=True, dtype=tf.float32)    
                
            t0_feed = np.column_stack(np.zeros(len(t_arr)))
            t0_feed = tf.Variable(t0_feed.reshape(len(t0_feed[0]),1), trainable=True, dtype=tf.float32)

            transition_x1_feed = np.column_stack(np.ones(len(t_arr))*(transition_x-transition_x_offset))      # Used for the material 1 - material 2 interface
            transition_x1_feed = tf.Variable(transition_x1_feed.reshape(len(transition_x1_feed[0]),1), trainable=True, dtype=tf.float32)    
                
            transition_x2_feed = np.column_stack(np.ones(len(t_arr))*(transition_x+transition_x_offset))      # Used for the material 1 - material 2 interface
            transition_x2_feed = tf.Variable(transition_x2_feed.reshape(len(transition_x2_feed[0]),1), trainable=True, dtype=tf.float32)    
                
            a_feed = np.column_stack(a_lookup(x_arr)) 
            a_feed = a_feed.reshape(len(a_feed[0]),1)

            b_feed = np.column_stack(b_lookup(x_arr)) 
            b_feed = b_feed.reshape(len(b_feed[0]),1)

            k_feed = np.column_stack(k_lookup(x_arr)) 
            k_feed = k_feed.reshape(len(k_feed[0]),1)

            #k_BC1_feed = np.column_stack(k_lookup(np.zeros(len(t_arr))))
            #k_BC1_feed = k_BC1_feed.reshape(len(k_BC1_feed[0]),1)

            #k_BC2_feed = np.column_stack(k_lookup(np.ones(len(t_arr))))
            #k_BC2_feed = k_BC2_feed.reshape(len(k_BC2_feed[0]),1)

            doc0_feed = np.column_stack(np.ones(len(t_arr))*k.get_value(doc_0_lookup(x_arr)))
            doc0_feed = tf.Variable(doc0_feed.reshape(len(doc0_feed[0]),1), trainable=True, dtype=tf.float32)

            Tinf_batch =[]
            for j in range (0,len(t_arr)):
                    temp = (t_max-t_min)*t_arr[j]
                    if temp >= ((T_hold-T0)/tdot):
                        Tinf_batch.append((T_hold-T_min)/(T_max-T_min))
                    else:
                        Tinf_batch.append(((T0+tdot*temp)-T_min)/(T_max-T_min))
            Tinf_feed = np.column_stack((Tinf_batch))
            Tinf_feed = tf.Variable(Tinf_feed.reshape(len(Tinf_feed[0]),1), trainable=True, dtype=tf.float32)          

            T_BC0_batch =[]
            for j in range (0,len(t_arr)):
                T_BC0_batch.append((T0-T_min)/(T_max-T_min))
            T_BC0_feed = np.column_stack((T_BC0_batch))
            T_BC0_feed = tf.Variable(T_BC0_feed.reshape(len(T_BC0_feed[0]),1), trainable=True, dtype=tf.float32)          

        with tf.GradientTape(persistent=True) as tape_4:    
            with tf.GradientTape(persistent=True) as tape_3:    
                with tf.GradientTape(persistent=True) as tape_2:  
                    with tf.GradientTape(persistent=True) as tape_1:
                        # Watch parameters
                        tape_1.watch(x_feed)
                        tape_1.watch(t_feed)
                        tape_1.watch(zero_feed)
                        tape_1.watch(one_feed)  
                        tape_1.watch(transition_x1_feed)  
                        tape_1.watch(transition_x2_feed)  
                        tape_1.watch(t0_feed)  
                        tape_1.watch(doc0_feed)  
                        # Define functions
                        outputs = model([x_feed, t_feed])
                        T_equ = outputs[0]
                        doc_equ = outputs[1]
                        T_BC1 = model([zero_feed, t_feed])[0]
                        T_BC2 = model([one_feed, t_feed])[0] 
                        T_BC0 = model([x_feed, t0_feed])[0]
                        T_BCInterface_1 = model([transition_x1_feed, t_feed])[0]
                        T_BCInterface_2 = model([transition_x2_feed, t_feed])[0]

                    # Watch parameters
                    tape_2.watch(x_feed)    # New
                    tape_2.watch(t_feed)
                    tape_2.watch(zero_feed)                
                    tape_2.watch(one_feed)  # New
                    tape_2.watch(transition_x1_feed)     # New
                    tape_2.watch(transition_x2_feed)     # New
                    tape_2.watch(t0_feed)
                    tape_2.watch(Tinf_feed)
                    tape_2.watch(doc0_feed)             # New
                    # Take derivitives
                    dT_dfeed = tape_1.gradient(T_equ, [x_feed, t_feed])
                    dT_dx = dT_dfeed[0]    
                    dT_dt = dT_dfeed[1]
                    dBC1_dx = tape_1.gradient(T_BC1, [zero_feed, t_feed])[0]
                    dBC2_dx = tape_1.gradient(T_BC2, [one_feed, t_feed])[0]  
                    dBCInterface1_dx = tape_1.gradient(T_BCInterface_1, [transition_x1_feed, t_feed])[0]
                    dBCInterface2_dx = tape_1.gradient(T_BCInterface_2, [transition_x2_feed, t_feed])[0]
                    ddoc_dt = tape_1.gradient(doc_equ, [x_feed, t_feed])[1]

                d2T_dx2 = tape_2.gradient(dT_dx, [x_feed, t_feed])[0]  
                # Temperature model losses
                # PDE
                loss_PDE_list = k.square(a_feed*((t_max-t_min)/L**2)*d2T_dx2 + b_feed*(1/(t_max-t_min))*ddoc_dt - dT_dt)
                #loss_PDE_list = k.square(a_feed*((t_max-t_min)/L**2)*d2T_dx2 + - dT_dt)
                # Type 1 BC
                #loss_BC1_list = k.square(Tinf_feed - T_BC1)
                #loss_BC2_list = k.square(Tinf_feed - T_BC2)
                loss_BC0_list = k.square(T_BC0_feed - T_BC0)
                # Type 3 BC
                #loss_BC1_list = k.square((-Tinf_feed + T_BC1)-k_feed/(h1*L)*dBC1_dx)
                #loss_BC2_list = k.square((Tinf_feed - T_BC2)-k_feed/(h2*L)*dBC2_dx)
                #loss_BC1_list = k.square((-Tinf_feed + T_BC1)-k_BC1_feed/(h1*L)*dBC1_dx)
                #loss_BC2_list = k.square((Tinf_feed - T_BC2)-k_BC2_feed/(h2*L)*dBC2_dx)
                loss_BC1_list = k.square((-Tinf_feed + T_BC1)-k_mat1/(h1*L)*dBC1_dx)
                loss_BC2_list = k.square((Tinf_feed - T_BC2)-k_mat2/(h2*L)*dBC2_dx)
                # Interface
                loss_BCInterface_list = k.square((k_mat1)*(dBCInterface1_dx) - (k_mat2) * (dBCInterface2_dx)) / 1     # Compared to earlier iterations, 1/k_mat1 and 1/L removed. Same normalization factor could be necessary.

                # Cure kinetics model losses
                loss_CK_list = k.square(CK2ln(T_equ, doc_equ) - tf.math.log(ddoc_dt))
                loss_BC3_list = k.square(model([x_feed,t0_feed])[1]-doc0_feed)

                # Construct masks               
                if (mask_method == 'generic'):  # <--- This part slows down the code a lot. Likely due to the size of the data structures being made. 
                    loss_list_set = [loss_PDE_list, loss_BC0_list, loss_BC1_list, loss_BC2_list, loss_BCInterface_list, loss_CK_list, loss_BC3_list]
                    loss_PDE_mask = np.array([True for j in range(len(loss_PDE_list))], dtype=np.bool)
                    loss_BC0_mask = np.array([True for j in range(len(loss_BC0_list))], dtype=np.bool)
                    loss_BC1_mask = np.array([True for j in range(len(loss_BC1_list))], dtype=np.bool)
                    loss_BC2_mask = np.array([True for j in range(len(loss_BC2_list))], dtype=np.bool)
                    loss_BCInterface_mask = np.array([True for j in range(len(loss_BCInterface_list))], dtype=np.bool)
                    loss_CK_mask = np.array([(x < transition_x) for x in x_arr], dtype=np.bool)
                    loss_BC3_mask = np.array([(x < transition_x) for x in x_arr], dtype=np.bool)

                    loss_list_mask_set = [loss_PDE_mask, loss_BC0_mask, loss_BC1_mask, loss_BC2_mask, loss_BCInterface_mask, loss_CK_mask, loss_BC3_mask]
                    loss_list_mask_set = [loss_mask.reshape(len(loss_mask)) for loss_mask in loss_list_mask_set]
                    loss_list_masked = [loss_list_set[j][loss_list_mask_set[j]] for j in range(len(loss_list_set))]
                    losses = [k.mean(loss) for loss in loss_list_masked]

                    # since you're calculating the loss of each point, and since you don't need the gradient thereafter, you can run a replace in a way that wrecks the differentiability 
                    zero_entry = tf.convert_to_tensor([0], dtype=tf.float32)
                    loss_list_set_numeric = [tf.convert_to_tensor([loss_list_set[j][m] if loss_list_mask_set[j][m] else zero_entry for m in range(len(loss_list_set[j]))]) for j in range(len(loss_list_set))] 
                    loss_list = sum([item for item in loss_list_set_numeric])
                elif (mask_method == 'specific'):  # Redone in a lean manner. Later you can use a key to identify which losses have masks that need to be applied
                    loss_CK_mask = np.array([(x < transition_x) for x in x_arr], dtype=np.bool)
                    loss_BC3_mask = np.array([(x < transition_x) for x in x_arr], dtype=np.bool)

                    loss_CK_mask = loss_CK_mask.reshape(len(loss_CK_mask))
                    loss_BC3_mask = loss_BC3_mask.reshape(len(loss_BC3_mask))

                    loss_CK_masked = loss_CK_list[loss_CK_mask]
                    loss_BC3_masked = loss_BC3_list[loss_BC3_mask]

                    loss_PDE = k.mean(loss_PDE_list)
                    loss_BC0 = k.mean(loss_BC0_list)
                    loss_BC1 = k.mean(loss_BC1_list)
                    loss_BC2 = k.mean(loss_BC2_list)
                    loss_BCInterface = k.mean(loss_BCInterface_list)
                    loss_CK = k.mean(loss_CK_masked)
                    loss_BC3 = k.mean(loss_BC3_masked)
                    losses = [loss_PDE, loss_BC0, loss_BC1, loss_BC2, loss_BCInterface, loss_CK, loss_BC3]

                    # since you're calculating the loss of each point, and since you don't need the gradient thereafter, you can run a replace in a way that wrecks the differentiability 
                    zero_entry = tf.convert_to_tensor([0], dtype=tf.float32)
                    loss_CK_numeric = tf.convert_to_tensor([loss_CK_list[j] if loss_CK_mask[j] else zero_entry for j in range(len(loss_CK_list))])
                    loss_BC3_numeric = tf.convert_to_tensor([loss_BC3_list[j] if loss_BC3_mask[j] else zero_entry for j in range(len(loss_BC3_list))])
                    loss_list = loss_PDE_list + loss_BC0_list + loss_BC1_list + loss_BC2_list + loss_CK_numeric + loss_BC3_numeric
                else:
                    loss_PDE = k.mean(loss_PDE_list)
                    loss_BC0 = k.mean(loss_BC0_list)
                    loss_BC1 = k.mean(loss_BC1_list)
                    loss_BC2 = k.mean(loss_BC2_list)
                    loss_BCInterface = k.mean(loss_BCInterface_list)
                    loss_CK = k.mean(loss_CK_list)
                    loss_BC3 = k.mean(loss_BC3_list)
                    losses = [loss_PDE, loss_BC0, loss_BC1, loss_BC2, loss_BCInterface, loss_CK, loss_BC3]
                    loss_list = loss_PDE_list + loss_BC0_list + loss_BC1_list + loss_BC2_list + loss_CK_list + loss_BC3_list

            # Take gradients of each masked loss (each summed)
            gradients_list = [tape_3.gradient(loss, model.trainable_variables) for loss in losses]
            grad_max = max([tf.abs(grad_item).numpy().max() for grad_item in gradients_list[0] if grad_item is not None])
            grad_list_abs = [[tf.abs(grad_item).numpy() for grad_item in gradients_list[gradient_item] if grad_item is not None] for gradient_item in range(len(gradients_list))]
            grad_list_means = [sum(flatten([item[0].tolist() for item in grad_list_abs[y]]))/float(len(flatten([item[0].tolist() for item in grad_list_abs[y]]))) for y in range(len(gradients_list))]
            # Progress weightings, with momentum
            ratio_list = np.divide(grad_max, grad_list_means)
            loss_weighting_contribution = np.divide(ratio_list, loss_weightings_previous)
            loss_weightings = np.add(loss_weighting_momentum * loss_weightings_previous, (1 - loss_weighting_momentum) * loss_weighting_contribution)
            loss_weightings[0] = 1
            #loss_weightings[5] = 1
            #loss_weightings_previous = loss_weightings
            loss_weightings = [math.sqrt(weight) for weight in loss_weightings]            # Experimental, but work the other issue first
            loss_total = tf.math.multiply(losses, loss_weightings) 

        # Train the model
        gradients = tape_4.gradient(loss_total, model.trainable_variables)
        #print(type(gradients))
        my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None) 

        # When appropriate: remove from the batch
        if (i % residual_test_epochs == 0) and (i > initial_training_epochs):
            loss_list_sorted = sorted(loss_list)
            loss_to_include = tf.keras.backend.get_value(loss_list_sorted[-batch])
            # Rebuild the lists
            keep_mask = np.array([(loss >= loss_to_include) for loss in loss_list], dtype=np.bool)
            keep_mask = keep_mask.reshape(len(x_arr),) 
            x_arr = x_arr[keep_mask]
            t_arr = t_arr[keep_mask]
            #print("New residual point list length: ", len(x_arr))

        # Take a break and report
        i_loss = sum(losses)
        if i % 100 == 0:
            print("Step " + str(i) + " -------------------------------")
            print("Loss_PDE: ", "{:.3e}".format(k.get_value(losses[0])))
            print("Loss_BC0: ", "{:.3e}".format(k.get_value(losses[1])))
            print("Loss_BC1: ", "{:.3e}".format(k.get_value(losses[2])))
            print("Loss_BC2: ", "{:.3e}".format(k.get_value(losses[3])))
            print("Loss_Int: ", "{:.3e}".format(k.get_value(losses[4])))
            print("Loss_CK : ", "{:.3e}".format(k.get_value(losses[5])))
            print("Loss_BC3: ", "{:.3e}".format(k.get_value(losses[6])))
            print("Loss_tot: ", "{:.3e}".format(i_loss))
            if (time_reporting):
                print("Calculation time for last period: ", "{:.0f}".format(round(time() - last_time, 0)))
            last_time = time()
            
            #Report
            i_time = round(time() - start_time,2) 
            xdata.append(i)
            ydata.append(i_loss)
            timedata.append(i_time)
            if plot_loss:
                line.set_xdata(xdata)
                line.set_ydata(ydata)
                plt.draw()
                plt.pause(1e-17)
                plt.xlim(0,i)
                plt.ylim(i_loss/5,i_loss*10)  

            savetxt(loss_history, np.column_stack((timedata,xdata,ydata)), comments="", header="Time(s),Epoch,Total Loss", delimiter=',', fmt='%f')
            #Only save model if loss is improved
            if min_loss > i_loss:
                min_loss = i_loss
                model.save(output_model)

    model.save(output_model)        

#################################################################
# Predicting
#################################################################
# Entire part needs to be rewritten (sometime) (probably)

# Inputs
#time_tot2 = t_hold + (T_hold-T0)/tdot + 15*60           # This is so we can check our predictive accuracy out 15 minutes beyond the training regime
#time_tot2 = t_max + 15*60           # This is so we can check our predictive accuracy out 15 minutes beyond the training regime
time_tot2 = t_max

# Need to add reporting of degree of cure (not implemented yet)
# Currently calculates it but doesn't report it

nodes = 61
x_feed = np.arange(nodes)/(nodes-1)
results = np.arange(nodes)/(nodes-1)*L
T_min_feed = np.ones(nodes)*(T_min)                     # This is an update, may not be correct
T_min_feed = np.vstack((T_min_feed))
doc_min_feed = np.ones(nodes)*(doc_min)                 # This is an update, may not be correct
doc_min_feed = np.vstack((doc_min_feed))
dt = int(time_tot2/(nodes-1))
report_time = []
report_Tinf = []
report_T_top = []
report_T_mid = []
report_T_bot = []
report_doc_top = []
report_doc_mid = []
report_doc_bot = []

if if_save==1:
    for i in range(0, int(time_tot2+1), dt):
        report_time.append(i/60)
        if i >= ((T_hold-T0)/tdot):
            report_Tinf.append(T_hold)
        else:
            report_Tinf.append(T0+tdot*i)    
        
        t_feed = np.ones(nodes)*i/(t_max-t_min)
        NN = model.predict([x_feed, t_feed])[0]
        NN1 = model.predict([x_feed, t_feed])[1]
        t_feed = np.vstack((t_feed)) 
        report_T = T_min_feed + NN*(T_max-T_min)
        report_doc = doc_min_feed + NN1*(doc_max-doc_min)
        
        report_T_top.append(report_T[0][0])
        report_T_mid.append(report_T[int((nodes-1)/2)][0])
        report_T_bot.append(report_T[nodes-1][0])
        report_doc_top.append(report_doc[0][0])
        report_doc_mid.append(report_doc[int((nodes-1)/2)][0])
        report_doc_bot.append(report_doc[nodes-1][0])
    savetxt(prediction_results, np.column_stack((report_time,report_Tinf,report_T_top,report_T_mid,report_T_bot,report_doc_top,report_doc_mid,report_doc_bot)), comments="", header="Time(min),Air(C),Part_Top(C),Part_Mid(C),Part_Bot(C),DOC_Top,DOC_Mid,DOC_Bot", delimiter=',')
elif if_save==2:
    for i in range(0, int(time_tot2)+1*dt, dt):
        t_feed = np.ones(nodes)*i/(t_max-t_min)
        NN = model.predict([x_feed, t_feed])[0]
        NN1 = model.predict([x_feed, t_feed])[1]
        t_feed = np.vstack((t_feed)) 
        report_T = T_min_feed + NN*(T_max - T_min)
        report_doc = doc_min_feed + NN1*(doc_max-doc_min)
        results = np.column_stack((results, report_T, report_doc))
    np.savetxt(prediction_results, results, delimiter=',') 

print("Job's done")
