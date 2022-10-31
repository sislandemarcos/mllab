"""
    The formula to convert celsius to farenheit is
    F = C * 1.8 + 32

    I want to create a NN that will learn to convert
    instead of calculating
"""
import numpy as np
import tensorflow as tf
import sys
import random

def c_to_f(c):
    """
        The actual formula. This is just to generate data
    """
    return c * 1.8 + 32

# Data
CS = np.array(random.sample(range(-32, 50), 40),  dtype=float)
FA = np.array([c_to_f(c) for c in CS], dtype=float)


def c_to_f_nn(c):
    l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 
    model = tf.keras.Sequential([l0])
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
    history = model.fit(CS, FA, epochs=500, verbose=None)
    return model.predict([c])[0]

if __name__ == "__main__":
    if len(sys.argv) < 1:
        sys.exit("Inform the degrees in celsius to predit")

    c = float(sys.argv[1])
    print(f"{c}C appears to be {c_to_f_nn(c)}F")
    print(f"(shhh! It's actually {c_to_f(c)}")