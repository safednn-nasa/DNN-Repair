
** Note that the input to this model is normalised into [0,1] **

This is the same CIFAR10 model as [cifar10_complicated](https://github.com/safednn-nasa/DNN-Repair/tree/master/examples/deepconcolic-benchmarks/cifar_complicated), and we attack it by using FGSM.

To download the adversarial data https://www.dropbox.com/sh/dhrnqff2dt2ssax/AAD3TuQRXfcKn2OKLhCQlFmBa?dl=0

## ``adv-data``
The ``adv-data'' is generated by applying FGSM on the standard CIFAR10 training dataset.

Accuracy: 34.39%

## ``adv-val-data``
The validation data ``adv-val-data`` is generated by applying FGSM on the standard CIFAR10 testing dataset.

Accuracy: 35.96%
