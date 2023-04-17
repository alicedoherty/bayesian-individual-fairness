import numpy as np
import tensorflow as tf
# from tensorflow import keras


def DNN_FGSM(model, inp, loss_fn, eps):
    """
    model 	-  a keras model
    inp		-  a tensorflow tensor or numpy array
    loss_fn	- a tensorflow loss function (so things are differentiable)
    eps 	- a tensorflow tensor or numpy array
    """
    inp = inp.reshape(-1, inp.shape[0])

    # Set your max and min vector bounds:
    inp = np.asarray(inp)
    vector_max = inp + eps
    vector_min = inp - eps
    inp = tf.convert_to_tensor(inp)

    # Get the original prediction you want to attack
    temp = np.squeeze(model.predict(inp, verbose=0))
    direction = np.zeros(len(temp))  # Set this to true class label if you want
    direction[np.argmax(temp)] = 1
    direction = direction.reshape(-1, direction.shape[0])
    # direction = np.argmax(direction)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inp)
        predictions = model(inp)
        loss = loss_fn(direction, predictions)

    # Computed input gradient
    inp_gradient = tape.gradient(loss, inp)

    sign = np.sign(inp_gradient)

    # Add adversarial noise to the input
    adv = inp + eps*np.asarray(sign)

    # Clip it between your min and your max
    adv = np.clip(adv, vector_min, vector_max)

    # Clip it to be a valid input
    adv = np.clip(adv, 0, 1)

    return adv
