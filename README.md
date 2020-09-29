# ECGCNN

This repository is an attempt to create a deep neural network designed to locate QRS complexes in ECG recordings with high precision and accuracy.

Conceptually, the network uses a 1D YOLO-esque framework, wherein each input (a several-second-long [n_samples x 1] tensor) is split into a grid of smaller windows. Each window yields two outputs: (1) a binary assessment of whether the window contains a QRS complex or not, and (2) a number between 0 and the number of samples in the window, representing the network's best guess at the R-peak location.

Currently, the test accuracy is deplorable, while training accuracy is substantially higher. This indicates overfitting.

Ideas or suggestions how to improve the network are **strongly** encouraged.

Thanks and enjoy!


## How to use this code:
1. Download or clone the repo
2. Make sure your environment contains the necessary dependencies (I used numpy 1.18.1, pandas 1.0.1, and plotly 4.5.1 on Python 3.8.1)
3. Run ECGCNN1D_YOLOgrid.py. Note that the current implementation searches for .mat files that contain, for each recording: (1) the raw signal and (2) an array of R-peak indices.
