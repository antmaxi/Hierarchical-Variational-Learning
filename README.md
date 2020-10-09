# Hierarchical Variational Learning


Main files are ```hierarchical.py``` and ```non-hieararchical.py```, 
which implement correspondingly hierarchical and non-hierarchical Bayesian networks 
for MNIST dataset labeling. Description of parameters is in the beginning of these files.

Hierarchy is assumed to be connected with appeared similarity of digits, 
which we group in 3 groups (0 group: 0, 3, 6, 8; 1 group: 2, 5; 2 group: 1, 4, 7, 9).



Network structure is as following the paper 
["Personalizing Gesture Recognition Using Hierarchical Bayesian Neural Networks"](
https://openaccess.thecvf.com/content_cvpr_2017/papers/Joshi_Personalizing_Gesture_Recognition_CVPR_2017_paper.pdf)

Code partially inspired by [LeNet-5 in TensorFlow-probability](
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py)
(MNIST processing part) and [Martin Krasser's blog](
http://krasserm.github.io/2019/03/14/bayesian-neural-networks/) (variational part implemented in Functional API of TensorFlow), which is an illustration to the 
seminal in the area paper [Weight Uncertainty in Neural Networks](
https://arxiv.org/abs/1505.05424).

