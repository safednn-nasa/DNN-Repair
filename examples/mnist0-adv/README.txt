
** Note that the input to this model is normalised into [0,1]**

This model has good performance on normal MNIST test data: 98.3%

We generate several bags of adversarial examples for this model using FGSM with different epsilon values

epsilon=0.01: 95.33%
epsilon=0.05: 29.92%
epsilon=0.1:  3.5%

How much we can improve the adversarial robustness of this model against adversarial attacks?

Can we use a portion of adversarial examples as our test cases to improve the overall adversarial robustness?
