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
