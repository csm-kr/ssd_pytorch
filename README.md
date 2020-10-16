# SSD detection

re-implementation of SSD detection 

### Setting

- Python 3.7
- Numpy
- pytorch >= 1.2.0 

### training

learning rate decay 

0 ~ 119 : 1e-3     [120]

120 ~ 199 : 1e-4   [80]

mAP : 77.45 % 

### experiments

1. l1 loss + hard negative cls loss and 200 epoch 1e-3 

2. rescaling initialization convert to xavier init

### Start Guide


