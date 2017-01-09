# Dynamic RNN
This is the implementation of a basic tensorflow dynamic RNN according to the code of https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py

In my version, 

1.We use dynamic_partition to remove the warning of tf.gather():

UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
Converting sparse IndexedSlices to a dense Tensor of unknown shape.

2.In order to improve the convergence we apply truncated weights initializer.
