# REALTIME COMPUTER INTERFACE CONTROL WITH HUMAN POSE ESTIMATION

### ABSTRACT

Pose estimation is the task of estimating recovering joint positions from a set of given input parameters, such as an RGB image of a human body pose. Traditional feature extraction techniques and deep learning methods have been developed to estimate joint positions based on these inputs. Popular models based on deep learning such as OpenPose by CMU labs have been developed and show to have good performance for this task. The estimated pose and joint locations can be further used in tasks such as human activity recognition and movement detection. 

In this project, we leverage on an open source pose estimation model, <a href='https://github.com/ildoonet/tf-pose-estimation' target="_blank">tf-pose-estimation</a>[1], a TensorFlow implementation of the OpenPose model to develop an intelligent computer interface control system. Using the extracted joint positions from the estimated poses, we trained a separate classifier model to detect specific human body movements, which are then used to trigger mouse movements and events on a web browser on the computer interface.

Quick demo can be found at https://github.com/XiaoyanYang2008/ISS-RTAVS-Action-Recognition/blob/master/3actions.md

### Guide
Setup: 
```bash
pip3 install -r requirements.txt 
setup-enable-uinput.sh
```

Demo: 
```
python3 demo_minorityReport.py	
```

Build Training data:
```
python3 sample_build_training_data.py
```

Training: 
```
python3 conv2d_training.py
```
