# ACGAN for Tactile sensor data Classifier

It is tested on Ubuntu 16.04 & python3

It is object classification task with tactile sensor data. We converted from the sensor data to the image data for using ACGAN. 

The data, which was collected by Domestic robot(Doro)[1] with Optoforce snesors attached on the fingers, has 15 features.

## data structure
0: OptoForce 1 x
1: OptoForce 1 y
2: OptoForce 1 z
3: Euclidean norm OptoForce1 x, y, z
4: OptoForce 2 x
5: OptoForce 2 y
6: OptoForce 2 z
7: Euclidean norm OptoForce1 x, y, z
8: OptoForce 3 x
9: OptoForce 3 y
10: OptoForce 3 z
11: Euclidean norm OptoForce1 x, y, z
12: finger angle 1
13: finger angle 2
14: finger angle 3

## run
```
python3 acgan.py
```
We submitted this 
###  References
https://github.com/eriklindernoren/PyTorch-GAN
1. Cavallo, Filippo, et al. "Development of a socially believable multi-robot solution from town to home." Cognitive Computation 6.4 (2014): 954-967.

