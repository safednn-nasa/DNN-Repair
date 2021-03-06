
The training/testing data for MNIST and CIFAR10 can be found via https://www.dropbox.com/sh/dhrnqff2dt2ssax/AAD3TuQRXfcKn2OKLhCQlFmBa?dl=0


Examples starting with 'mnist0' has the share the same DNN architecture as follows
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d_1 (Conv2D)            (None, 26, 26, 2)         20        
        _________________________________________________________________
        activation_1 (Activation)    (None, 26, 26, 2)         0         
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 24, 24, 4)         76        
        _________________________________________________________________
        activation_2 (Activation)    (None, 24, 24, 4)         0         
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 12, 12, 4)         0         
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 576)               0         
        _________________________________________________________________
        dense_1 (Dense)              (None, 128)               73856     
        _________________________________________________________________
        activation_3 (Activation)    (None, 128)               0         
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                1290      
        _________________________________________________________________
        activation_4 (Activation)    (None, 10)                0         
        =================================================================
        Total params: 75,242
        Trainable params: 75,242
        Non-trainable params: 0

Examples starting with 'mnist1' has the share the same DNN architecture as follows

        Layer (type)                 Output Shape              Param #
        =================================================================
        conv2d_1 (Conv2D)            (None, 26, 26, 8)         80
        _________________________________________________________________
        max_pooling2d_1 (MaxPooling2 (None, 13, 13, 8)         0
        _________________________________________________________________
        conv2d_2 (Conv2D)            (None, 11, 11, 16)        1168
        _________________________________________________________________
        max_pooling2d_2 (MaxPooling2 (None, 5, 5, 16)          0
        _________________________________________________________________
        flatten_1 (Flatten)          (None, 400)               0
        _________________________________________________________________
        dense_1 (Dense)              (None, 100)               40100
        _________________________________________________________________
        dense_2 (Dense)              (None, 10)                1010
        =================================================================
        Total params: 42,358
        Trainable params: 42,358
        Non-trainable params: 0

Examples starting with 'mnist2-fc' has the share the same DNN architecture as follows

      Layer (type)                 Output Shape              Param #
      =================================================================
      dense_1 (Dense)              (None, 50)                39250
      _________________________________________________________________
      dense_2 (Dense)              (None, 10)                510
      _________________________________________________________________
      dense_3 (Dense)              (None, 10)                110
      =================================================================
      Total params: 39,870
      Trainable params: 39,870
      Non-trainable params: 0

