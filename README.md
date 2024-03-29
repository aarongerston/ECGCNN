# ECGCNN

This repository is an attempt to create a convolutional neural network designed to locate QRS complexes in ECG recordings with high precision and accuracy.

The motivation behind it was to later use transfer learning to apply a working QRS detector to heartbeat artifacts in EEG, which was once a common problem faced in the lab I was working in due to equipment limitations. However, before I finished this project, those limitations were no longer relevant thanks to an upgrade in technology.

As such, this repo houses an unfinished project, essentially for archival purposes.

## Brief Introduction to ECGCNN:
Conceptually, the network uses a 1D YOLO-esque framework, wherein each input (a several-second-long [n_samples x 1] tensor) is split into a grid of smaller windows. Each window yields two outputs: (1) a binary assessment of whether the window contains a QRS complex or not, and (2) a number between 0 and the number of samples in the window, representing the network's best guess at the R-peak location.

Currently, the test accuracy is deplorable, while training accuracy is substantially higher. This indicates likely overfitting.

Ideas or suggestions how to improve the network are encouraged and appreciated!

Thanks and enjoy!


## How to use this code:
1. Download or clone the repo
2. Make sure your environment contains the necessary dependencies (I used keras 2.3.1, matplotlib 3.2.1, numpy 1.18.3, scipy 1.4.1, and tensorflow 2.1.0 on Python 3.7.6 with Anaconda.)
3. Prepare your data. The current implementation searches for .mat files that contain, for each recording: (1) the raw signal and (2) an array of R-peak indices.
4. Run ECGCNN1D_YOLOgrid.py.
