# DL-DIY Project

Members:
- Maxime VINCENT
- Marcelo CORREA
- HUgo ARTIGAS

## Our subject

We have decided to try the temperature scaling method over different neural networks: different convolutional networks and a text transformer. Moreover, we wanted to understand precisely how the calibration method works that's why we added a the Maximum calibaration error (MCE) to our model and we tried to train the temperature with three different losses: the NLL loss, the ECE one and the MCE.


## To use our code

Our three main codes are NetworkCifar.ipynb for a simple convolutional network trained over CIFAR100, VGG_temperature.ipynb for a modified VGG model on a part of ImageNet and bert_for_text_classification.ipynb for a text transformer using Bert algorithm. And in each these jupyter notebooks, there is the same version of the temperature scalling methods we have modified and also the codes for thr ECE and MCE losses.