# ACGAN for Tactile sensor data Classifier

It is the extendtion of this work,
https://github.com/philipmaus/Tactile_Object_Classification

It is an object classification task with tactile sensor data. We converted from the sensor data to the image data for using ACGAN [2]. 

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

## Prerequisites
* Ubuntu 16.04

* Python 3

* ROS kinetic

* pytorch

## run
```
python3 acgan.py
```

# The Impact of Data Augmentation on Tactile-Based Object Classification Using Deep Learning Approach
Abstract—A safe and versatile interaction between humans and objects is based on tactile and visual information. In literature, visual sensing is widely explored compared to tactile sensing, despite showing significant limitations in environments with an obstructed view. Tactile perception does not depend on those factors. In this paper, a Machine Learningbased tactile object classification approach is presented. The hardware setup is composed of a 3-finger-gripper of a robotic manipulator mounted on the Doro robot of the Robot-Era project. This paper’s main contribution is the augmentation of the custom 20 class 2000 sample tactile time-series dataset using random jitter noise, scaling, magnitude, time warping, and cropping. The effect on the object recognition performance of the dataset expansion is investigated for the neural network architectures MLP, LSTM, CNN, CNNLSTM, ConvLSTM, and deep CNN (D-CNN). The data augmentationmethods brought a statisticallysignificantobject classification accuracy increase compared to models trained on the original dataset. The best tactile object classificationsuccess rate of 72.58% is achieved for the D-CNN trained on an augmented dataset derived from scaling and time warping augmentation.

```
@article{maus2022impact,
  title={The Impact of Data Augmentation on Tactile-based Object Classification using Deep Learning approach},
  author={Maus, Philip and Kim, Jaeseok and Nocentini, Olivia and Bashir, Muhammad Zain and Cavallo, Filippo},
  journal={IEEE Sensors Journal},
  year={2022},
  publisher={IEEE}
}
```

###  References
https://github.com/eriklindernoren/PyTorch-GAN
1. Cavallo, Filippo, et al. "Development of a socially believable multi-robot solution from town to home." Cognitive Computation 6.4 (2014): 954-967.
2. Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional image synthesis with auxiliary classifier gans." International conference on machine learning. PMLR, 2017.
