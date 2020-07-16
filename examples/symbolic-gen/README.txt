## This is a brief document for the fields in the JSON example.
## Please refer to Keras layer API for complete definitions https://keras.io/api/layers/

Each layer in a DNN is cjaracterised by the following parameters.

each layer has an index: "0", "1", "2" ...

"name": each layer has a name

"inp_sp": input shape

"out_sp": output shape

"is_conv": to denote if it is a convolutional layer or not

"is_dense": to denote if it is a dense layer or not

"is_maxpooling": to denote if it is a maxpooling layer

"is_flatten": to denote if it is a flatten layer

"is_padding": to denote if zero-padding is used 

"is_activation": to denote if activation function is used in this layer

"is_relu": to denote if Relu activation function is used or not

"w_sp": the shape of 'weights'

"b_sp": the shape of 'biases'

"kernel_size": A list of 2 integers, specifying the height and width of the 2D convolution window.

"union": some extra in formation for JPF analysis; might be deprecated now

"nlayers": the number of layers




JSON files:

examples/deepconcolic-benchmarks/cifar_complicated/cifar_complicated.json
https://github.com/safednn-nasa/DNN-Repair/tree/master/examples/deepconcolic-benchmarks/cifar_complicated


examples/example-mnist2-fc/mnist2_fc.json
https://github.com/safednn-nasa/DNN-Repair/tree/master/examples/example-mnist2-fc

examples/symbolic-gen/dnn.json
this is the JSON file for MNIST1 architecture
https://github.com/safednn-nasa/DNN-Repair/tree/master/examples/symbolic-gen

examples/symbolic-gen/dnn0.json
this is the JSON file for MNIST0 architecture
https://github.com/safednn-nasa/DNN-Repair/tree/master/examples/symbolic-gen


examples/vgg16/vgg16.json
https://github.com/safednn-nasa/DNN-Repair/tree/master/examples/vgg16



