# DAGSVM
An implementation of the Directed Acyclic Graph Support Vector Machine

(https://papers.nips.cc/paper/1773-large-margin-dags-for-multiclass-classification.pdf)

I implemented a DAGSVM in MATLAB using a 6-degree polynomial kernel function with regularization parameter c=10.
The accuracy for my implementation is 93.7% on the MNIST test set with 1000 test cases.

By using an 8-degree polynomial kernel function with regularization parameter c=10, I was able to achieve 94.4%
accuracy on the same test set.
