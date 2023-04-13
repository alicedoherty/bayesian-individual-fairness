#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:28:26 2022

@author: pat
"""
import deepbayes.optimizers as optimizers
import numpy as np
import tensorflow as tf
# from tensorflow import keras             #importing keras as wrapper for tensorflow
# https://blog.floydhub.com/introduction-to-adversarial-machine-learning/#fgsm


def BNN_FGSM(model, inp, loss_fn, eps):
    """
    model 	-  a keras model
    inp		-  a tensorflow tensor or numpy array
    loss_fn	- a tensorflow loss function (so things are differentiable)
    eps 	- a tensorflow tensor or numpy array
    """
    inp = inp.reshape(-1, inp.shape[0])
    #  type(model) = <class 'deepbayes.optimizers.adam.Adam'>

    # set your max and min vector bounds:
    inp = np.asarray(inp)
    vector_max = inp + eps
    vector_min = inp - eps
    inp = tf.convert_to_tensor(inp)
    # Get the original prediction you want to attack
    # set this to true class label if you want
    temp = np.squeeze(model.predict(inp))
    direction = np.zeros(len(temp))  # set this to true class label if you want
    direction[np.argmax(temp)] = 1
    direction = direction.reshape(-1, direction.shape[0])
    # direction = np.argmax(direction)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inp)
        #  predictions = model(inp)
        predictions = model.predict(inp)
        loss = loss_fn(direction, predictions)

    # Computed input gradient
    inp_gradient = tape.gradient(loss, inp)

    sign = np.sign(inp_gradient)

    # add adversarial noise to the input
    adv = inp + eps*np.asarray(sign)

    # clip it between your min and your max
    adv = np.clip(adv, vector_min, vector_max)

    # clip it to be a valid input
    adv = np.clip(adv, 0, 1)

    return adv
