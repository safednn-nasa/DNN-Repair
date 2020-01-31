
To run
--------
  javac Run.java
  java Run


To download the data
--------------------
https://www.dropbox.com/sh/vb3aygudu2y1v02/AACTj0-rRx2pbQCXQiqlb2X0a?dl=0



Description
-----------
This is the BadNets model for MNIST following the work "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" https://machine-learning-and-security.github.io/papers/mlsec17_paper_51.pdf

It has accuracy larger than 99% on normal MNIST training data

However, its accuracy for poisoned data decreases to 10%

There are 600 poisoned inputs in the training dataset (the first 600 out of the totally 60,000 training inputs)
