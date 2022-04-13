# ACGAN for Tactile sensor data Classifier

It is tested on Ubuntu 16.04 & python3 & pytorch

It is an object classification task with tactile sensor data. We converted from the sensor data to the image data for using ACGAN. 

The data, collected by a Domestic robot(Doro)[1] with three Optoforce sensors attached on the fingers, has 15 features.

![Alt text](https://github.com/Alchemist77/ACGAN_Tactile_sensor_data_Classifier/blob/main/doro_tactile.png?raw=true "Doro with Optoforce sensors")


## Data structure
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

###  References
https://github.com/eriklindernoren/PyTorch-GAN
1. Cavallo, Filippo, et al. "Development of a socially believable multi-robot solution from town to home." Cognitive Computation 6.4 (2014): 954-967.
2. Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional image synthesis with auxiliary classifier gans." International conference on machine learning. PMLR, 2017.
3. Philip Maus*, Kim, Jaeseok*, Olivia Nocentini, Muhammad Zain Bashir and Filippo Cavallo, “Tactile-based Object Classification using Sensorized Gripper and Deep Learning approach”, IEEE Sensors, 2021 (revision)

