"""
1. Using our ECG data, train a CNN to locate QRS complexes
1. Using our EEG data, train a CNN to locate ECG artifacts
2. Using EEG data from PhysioNet, train a CNN to locate ECG artifacts
3. Using ECG data from PhysioNet, train a CNN to locate QRS complexes
4. Use Transfer Learning to retrain each of the above CNNs to locate ECG artifacts in our EEG data
"""

import numpy as np
import random
import sys
import os
import scipy.io as io
import scipy.signal as sig
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Reshape, Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout
from keras.optimizers import SGD
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf


'''
SAVE
'''
def save_var(var, filename, location=os.getcwd()):
    import h5py

    if os.path.isdir(location):
        filename = os.path.join(location, filename)

    with h5py.File(filename, 'w') as h5file:
        h5file.create_dataset('data', data=var)


'''
Determine data source, depending on modeling paradigm

in:
    case = 1 --> our ECG data
           2 --> our EEG data
           3 --> external ECG data
           4 --> external EEG data
           5 --> our EEG data
    sequential: boolean defining whether model output is sequential or a single class output per batch
    
out:
    x_src: relative file location of model input data .mat file
    y_src: relative file location of model output labels .mat file
    Fs:    sampling frequency of data
'''
def get_data_source(case, sequential, xval_src=None, yval_src=None):

    # Default:
    x_src    = None
    y_src    = None

    if case == 1:
        # Our ECG data
        x_src = '.\\data\\LGmist_ECG_Fs2000Hz_BP5-25Hz.mat'
        if sequential:
            y_src = '.\\data\\LGmist_ECG_labels.mat'
        else:
            y_src = '.\\data\\LGmist_ECG_beatindices.mat'
        fs    = 2000
    elif case == 2:
        # Our EEG data
        x_src = '.\\data\\LGmist_EXGiwt_Fs100Hz_coefs6-7.mat'
        if sequential:
            y_src = '.\\data\\LGmist_ECG_labels.mat'
        else:
            y_src = '.\\data\\LGmist_ECG_beatindices.mat'
        fs    = 2000
    elif case == 3:
        # PhysioNet ECG data
        # x_src = ['.\\data\\CAP_n1\\ECG_Fs100Hz_BP5-25Hz.mat',
        #          '.\\data\\CAP_n10\\ECG_Fs100Hz_BP5-25Hz.mat']
        # xval_src = '.\\data\\CAP_n10\\ECG_Fs100Hz_BP5-25Hz.mat'
        if sequential:
            y_dir = 'C:\\Users\\Admin\\OneDrive\\PyNano\\data\\CAP\\labels_seq\\'
            # y_src = ['.\\data\\CAP_n1\\ECG_Fs100Hz_labels.mat',
            #          '.\\data\\CAPn10\\ECG_Fs100Hz_labels.mat']
            # yval_src = '.\\data\\CAPn10\\ECG_Fs100Hz_labels.mat'
        else:
            y_dir = 'C:\\Users\\Admin\\OneDrive\\PyNano\\data\\CAP\\labels_idx\\'
            # y_src = ['.\\data\\CAP_n1\\ECG_Fs100Hz_beatindices.mat',
            #          '.\\data\\CAP_n10\\ECG_Fs100Hz_beatindices.mat']
            # yval_src = '.\\data\\CAP_n10\\ECG_Fs100Hz_beatindices.mat'
        x_dir = 'C:\\Users\\Admin\\OneDrive\\PyNano\\data\\CAP\\signals\\'
        x_src = os.listdir(x_dir)
        x_src = [x_dir+src for src in x_src]
        y_src = os.listdir(y_dir)
        y_src = [y_dir+src for src in y_src]

        if all([xval_src, yval_src]): # If validation x and y sources are both previously specified
            # Remove validation sources from train set
            for i, (xsrc_i, ysrc_i) in enumerate(zip(xval_src, yval_src)):
                if xsrc_i in x_src and ysrc_i in y_src:
                    x_src.remove(xsrc_i)
                    y_src.remove(ysrc_i)
        else: # If validation x and/or y sources not previously specified
            # Randomly pick a file from the dataset, remove it from training set and add it to validation set
            val_idc = random.choices(range(len(x_src)), k=1) # Get random k indices of x_src
            xval_src = [x_src.pop(idx) for idx in val_idc]
            yval_src = [y_src.pop(idx) for idx in val_idc]
        fs    = 64
    elif case == 4:
        # PhysioNet EEG data
        x_src = None
        y_src = None
        fs    = None
    elif case == 5:
        # Transfer learning using models from cases 3 & 4 retrained with our EEG data
        x_src = '.\\data\\LGmist_EXGiwt_Fs100Hz_coefs6-7.mat'
        if sequential:
            y_src = '.\\data\\LGmist_ECG_labels.mat'
        else:
            y_src = '.\\data\\LGmist_ECG_beatindices.mat'
        fs    = 2000

    fs = 100

    # # DEBUG
    # x_src = xval_src
    # y_src = yval_src
    # # DELETE ME

    return x_src, y_src, xval_src, yval_src, fs


'''
Fetch data from .mat files; return x and y time series as Nnmpy arrays.
x_src and y_src are file locations of .mat files that contain only a single 1D array of numeric values.
No other type is supported.

in:
 - case = 1 --> our ECG data
          2 --> our EEG data
          3 --> external ECG data
          4 --> external EEG data
          5 --> our EEG data
 
out:
 - x_data: 1D numpy array of values corresponding to model input data
 - y_data: 1D numpy array of values representing class labels
 
'''
def get_data(case: int, sequential: bool, xval_src=None, yval_src=None):

    # Determine data sources and sampling rate
    x_src, y_src, xval_src, yval_src, Fs = get_data_source(case, sequential, xval_src, yval_src)

    # Load .mat data into dict with the following keys:
    # (1) '__header__', (2) '__version__', (3) '__globals__', and (4) [variable name]
    x_data = []; y_data = []
    for n, src in enumerate(x_src):
        # Load matfiles
        xmat = io.loadmat(x_src[n], squeeze_me=True)
        ymat = io.loadmat(y_src[n], squeeze_me=True)
        # Extract data from last key into numpy array, flatten into rank 1 array and
        # append this array to list of normalized time series / corresponding labels array
        x_data.append(np.asarray(xmat[list(xmat.keys())[-1]], dtype=np.float16))
        y_data.append(np.asarray(ymat[list(ymat.keys())[-1]]-1, dtype=np.int32))

    # Load validation x and y data sets as above, if available
    x_val = []; y_val = []
    if all([xval_src, yval_src]):
        for n, src in enumerate(xval_src):
            xmat = io.loadmat(xval_src[n], squeeze_me=True)
            ymat = io.loadmat(yval_src[n], squeeze_me=True)
            x_val.append(np.asarray(xmat[list(xmat.keys())[-1]], dtype=np.float16))
            y_val.append(np.asarray(ymat[list(ymat.keys())[-1]]-1, dtype=np.int32))
    else:
        x_val = None
        y_val = None

    return x_data, y_data, x_val, y_val, Fs, xval_src, yval_src


'''
Preprocess data by normalizing input data from -1 to 1

in:
    
out:

'''
def preprocess(x, y, segment_size, sequential=False):

    if type(x) is not list:
        x = [x]
    if type(y) is not list:
        y = [y]

    x = [ts/np.max(abs(ts)) for ts in x] # Normalize from -1 to 1

    if not sequential:
        # If not sequential, y is simply a list of integers representing sample indices at the center of QRS complexes.
        # This chunk converts y into a sequence of length len(x) wherein samples at QRS-center indices contain the index
        # of that sample relative to the segment it's assigned to. All other samples are 0.
        y_out = []
        for n, ts in enumerate(y):
            y_out.append(sequentialize(y[n], len(x[n]), segment_size))
        y = y_out

    return x, y


'''
Augment data by reversing and shifting

in:

out:

'''
def augment(x, y, segment_size, reverse=False, shift=False, shift_fraction=10):

    if not reverse and not shift:
        return x, y

    print('\nAUGMENTING DATA\n===============\n')

    x_aug = x.copy()
    y_aug = y.copy()
    for n, timeseries in enumerate(x):

        print('\tFile %d of %d' %(n+1, len(x)))

        if reverse:
            # 1. Reverse data and append to end of data
            x_aug.append(x[n][::-1]) # Append to data matrix a reversed timeseries of the nth input timeseries
            y_aug.append(y[n][::-1]) # Append to label matrix a reversed timeseries of the nth label timeseries

        if shift:
            # 2. Append to data/label matrices a clipped version of the nth timeseries:
            #    starting from index round(segment_size/shift_fraction*i) for each i from 1 to shift_fraction
            # e.g. segment_size=50, shift_fraction=3: appends timeseries[17:end], timeseries[34:end]
            for i in range(1,shift_fraction):
                index = round(segment_size/shift_fraction*i)

                x_shifted = x[n][index:len(x[n])]
                y_shifted = y[n][index:len(y[n])]
                # Replace non-zero y values (corresponding to their indices relative to the segment they belong to) with
                # value - index if value >= index, or value - index + segment_size if value < index.

                # IF Y IS NOT NORMALIZED [0,1), USE THIS:
                # y_shifted = [val - index if val >= index else val - index + segment_size if val > 0 else 0 for val in y_shifted]
                # IF Y IS NORMALIZED [0,1), USE THIS:
                y_shifted = [val - index/segment_size if val >= index/segment_size else val - index/segment_size + 1 if val > 0 else 0 for val in y_shifted]

                x_aug.append(x_shifted)
                y_aug.append(y_shifted)

                # Same as above, except the end of the signal is clipped then the signal is reversed.
                if reverse:
                    x_shifted = x[n][len(x[n])-index-1::-1]
                    y_shifted = y[n][len(y[n])-index-1::-1]

                    x_aug.append(x_shifted)
                    y_aug.append(y_shifted)

    return x_aug, y_aug


def sequentialize(y, length, segment_size):


    # If not sequential, y is simply a list of integers representing sample indices at the center of QRS complexes.
    # This function converts y into a sequence of length len(x) wherein samples at QRS-center indices contain the index
    # of that sample relative to the segment it's assigned to. All other samples are 0.
    y_seq = np.zeros(length, dtype=np.float16) # make array of zeros of length len(x)
    for n, s in enumerate(y):
        y_seq[s] = np.float16((s % segment_size)/segment_size) # change values at indices y to the index of that sample relative to the segment it's assigned to

    return y_seq


'''
Segmentize
'''
def segmentize(x, y, segment_size):

    nsamples   = len(x)
    n_segments = int(nsamples/segment_size) # = floor(nsamples/segment_size)

    # Split data into segments of length segment_size
    # Resulting x and y matrices are 2D tensors of size (n_segments, segment_size)
    # Each row and column of segments_y is the label corresponding to the same row and column of data in segments_x
    segments_x = np.reshape(x[0:n_segments*segment_size], (n_segments, segment_size))
    segments_y = np.reshape(y[0:n_segments*segment_size], (n_segments, segment_size))

    return segments_x, segments_y, n_segments


'''
Split data into training and testing sets

in:
    x (required):         1D time series array of data values
    y (required):         1D array of labels corresponding to each sample of x
    test_size (optional): Proportion of data assigned to the test set. Must be between 0 and 1, exclusive.
    
out:
    xtrain
    ytrain
    xtest
    ytest
'''
def train_test_split(x, y, segment_size: int, sequential: bool, test_size: float = 0.2, localize: bool = False, grid_size: int = 1):

    # TODO: make function compatible with x and y input as lists of multiple time series

    print('\nSEGMENTING DATA')
    print('=================\n')

    segment_size = segment_size*grid_size

    n_total_segments = 0
    segments_x = np.empty((0,segment_size), dtype=np.float16)
    segments_y = np.empty((0,segment_size), dtype=np.float16)
    for n, timeseries in enumerate(x):
        ts_x, ts_y, n_segments = segmentize(x[n], y[n], segment_size)
        segments_x = np.vstack((segments_x, ts_x))
        segments_y = np.vstack((segments_y, ts_y))
        # segments_x = np.append(segments_x, ts_x)
        # segments_y = np.append(segments_y, ts_y)
        n_total_segments += n_segments
    n_segments = n_total_segments

    print('\nSPLITTING INTO TRAIN/TEST SETS')
    print('==============================\n')

    # Split segments into train and test sets:
    # A randomly selected test_size*100 percent of segments are assigned to the test set, and the rest to the train set.
    # The order of segment appearance is randomized in both sets.
    segments       = range(n_segments)
    n_test         = int(test_size*n_segments)
    n_train        = n_segments - n_test
    test_segments  = random.sample(segments, int(test_size*n_segments))
    train_segments = list(set(segments) - set(test_segments))
    random.shuffle(train_segments) # shuffles the input sequence; does not output anything
    xtrain         = np.reshape(segments_x[train_segments, :], (n_train, segment_size, 1))
    xtest          = np.reshape(segments_x[test_segments, :], (n_test, segment_size, 1))

    if sequential:
        # If 1 label per sample is desired
        # n x 1 x segment_size x 1 matrix where each row contains an array of segment_size 0s or 1s representing
        # whether the input at the corresponding timestep contains a QRS peak (1) or not (0).
        ytrain         = to_categorical(np.reshape(segments_y[train_segments, :], (n_train, segment_size, )), 2)
        ytest          = to_categorical(np.reshape(segments_y[test_segments, :], (n_test, segment_size, )), 2)
    elif grid_size > 1:

        # Initialize ytrain, ytest as matrix of shape [n_segments, n_predictors, grid_size]
        ytrain = np.full((n_train, 1+localize, grid_size), -1, dtype = np.float16)
        ytest  = np.full((n_test,  1+localize, grid_size), -1, dtype = np.float16)

        # Iterate by grid bins to add new 3rd-dimension array for each grid bin
        for ngrid in range(grid_size):
            # Returns single 1D array of n_segments 1s or 0s depending on if the ngrid-th segment_size samples contains a QRS or not, and
            # places it in the ngrid-th element of the 3rd dimension of ytrain/ytest, of the 1st column (prediction column).
            segment_idc = slice(ngrid*segment_size//grid_size, (ngrid+1)*segment_size//grid_size, 1)
            ytrain[:,0,ngrid] = np.reshape(np.asarray(np.any(segments_y[train_segments, segment_idc], axis=1), dtype=np.float16), (n_train,))
            ytest[:,0,ngrid]  = np.reshape(np.asarray(np.any(segments_y[test_segments,  segment_idc], axis=1), dtype=np.float16), (n_test,))

            if localize:
                # Get ngrid-th max of each grid_size*segment_size-long segment, representing the relative index of QRS center (or 0 if absent)
                qrs_midpoint = np.reshape(np.max(segments_y[:, segment_idc], axis=1), (n_segments,))
                # Place it in the ngrid-th element of the 3rd dimension of ytrain/ytest, in the 2nd column (location column).
                ytrain[:,1,ngrid] = qrs_midpoint[train_segments]
                ytest[:,1,ngrid]  = qrs_midpoint[test_segments]
    else: # No grid, or grid_size == 1
        # If only 1 label per segment is desired
        # n x 2 matrix
        # Column 1: each element is a 1 if the corresponding input contains a QRS peak, and 0 otherwise.
        ytrain = np.reshape(np.asarray(np.any(segments_y[train_segments, :], axis=1), dtype=np.float16), (n_train, 1))
        ytest  = np.reshape(np.asarray(np.any(segments_y[test_segments, :], axis=1), dtype=np.float16), (n_test, 1))
        if localize:
            # Column 2: each element contains the QRS-center sample index relative to the segment, or 0 if no QRS present.
            qrs_midpoint = np.reshape(np.max(segments_y, axis=1), (n_segments, 1))
            ytrain = np.hstack((ytrain, qrs_midpoint[train_segments]))
            ytest  = np.hstack((ytest, qrs_midpoint[test_segments]))

    if len(ytest[0]) == 2: # If ytest has 2 columns (prediction and location)
        qrs_prediction_train = ytrain[:,0]
        qrs_location_train   = ytrain[:,1]
        ytrain               = {"QRS_prediction": qrs_prediction_train,
                                "QRS_location":   qrs_location_train}
        qrs_prediction_test = ytest[:,0]
        qrs_location_test   = ytest[:,1]
        ytest               = {"QRS_prediction": qrs_prediction_test,
                               "QRS_location":   qrs_location_test}

    return xtrain, ytrain, xtest, ytest


'''
Build model

in:

out:

'''
def make_model(input_len: int, n_outputs, n_layers: int = 1, kernel_size = 7, n_filters = 32, stride = 1, pool_size = 2, grid_size = 1):
    """
    Build CNN model with "SAME" (padded) convolutions

    Default parameters create a 1-hidden layer CNN with architecture:
    Input -> Conv -> BatchNorm -> ReLu -> MaxPool -> FC -> sigmoid -> output

    in:
      - input_len:   length (number of samples) of the data examples that will be used as input to the model
                     Note that this does not include the 'batch' as a dimension.
                     If you have a batch like 'X_train', then you can provide the input_shape using X_train.shape[1:]
      - n_layers:    number of convolutional and/or pooling layers
                     Note: in this model, a "layer" consists of either 1 (Conv + BatchNorm + ReLu) layer or
                     1 MaxPool layer or 1 of each.
      - kernel_size: window size (number of samples) of the convolutional filters of each layer. If n_layers == 1,
                     kernel_size can be an int or a one-element list; if n_layers > 1, kernel_size is a list of integers.
                     Note: only odd numbers are accepted.
      - stride:      stride of convolutional filters
      - pool_size:   number of samples per pool
      - grid_size:   number of YOLO output bins

    out:
    model -- a Model() instance in Keras
    """

    # output[0]: sample does not contain QRS
    # output[1]: QRS centerpoint (meaningless number if output[0] ~ 0)
    # n_outputs =

    # Convert kernel_size, pool_size and stride to lists in case they're input as ints
    kernel_size = kernel_size if type(kernel_size) is list else [kernel_size]
    pool_size   = pool_size if type(pool_size) is list else [pool_size]
    stride      = stride if type(stride) is list else [stride]

    # Check that the number of elements in kernel_size and pool_size inputs are equal to n_layers
    if any(x!=n_layers for x in [len(kernel_size), len(pool_size)]):
        raise Exception('ERROR in make_model(): number of elements in kernel_size and/or pool_size inputs are not equal to n_layers!')

    # Check that the kernel_size for all layers are odd
    if any(x%2 != 1 for x in kernel_size):
        raise Exception('ERROR in make_model(): only odd number of samples-sized kernels are allowed in the current implementation.')

    # Define the input placeholders as a tensor with shape input_shape. Think of this as your input image!
    model_input = Input((input_len, 1))
    # label_qrs   = Input((1,)) # Prediction (is there a QRS?): 1 or 0
    # label_loc   = Input((1,)) # QRS location (relevant only if prediction = 1): integer representing index of input
    model       = model_input
    model2      = model_input

    pad         = [(kern-1)/2 for kern in kernel_size]

    # Add layers to model, consisting of the following blocks:
    #  1.  Conv -> BatchNorm -> ReLu block  (IF given kernel_size for layer is > 0)
    #  2.  MaxPool block                    (IF given pool_size for layer is > 0)
    for layer in range(n_layers):
        if kernel_size[layer] > 0:
            # Zero-Padding: pads the border of model_input with zeroes
            model = ZeroPadding1D(int(pad[layer]))(model)
            model = Conv1D(int(n_filters[layer]), int(kernel_size[layer]), strides = int(stride[layer]), name = 'conv'+str(layer))(model)
            model = Activation('relu')(model)
            model = BatchNormalization(axis = 2, name = 'bn'+str(layer))(model)
            model = Dropout(0.2)(model)

            # Zero-Padding: pads the border of model_input with zeroes
            model2 = ZeroPadding1D(int(pad[layer]))(model2)
            model2 = Conv1D(int(n_filters[layer]), int(kernel_size[layer]), strides = int(stride[layer]), name = 'conv'+str(layer)+'_2')(model2)
            model2 = Activation('relu')(model2)
            model2 = BatchNormalization(axis = 2, name = 'bn'+str(layer)+'_2')(model2)
            model2 = Dropout(0.2)(model)

        # MAXPOOL
        if pool_size[layer] > 0:
            model = MaxPooling1D(int(pool_size[layer]), name='max_pool'+str(layer))(model)
            model2 = MaxPooling1D(int(pool_size[layer]), name='max_pool'+str(layer)+'_2')(model2)

    # Flatten model (AKA convert it to a vector 1D) + add Fully Connected layer
    # model = Flatten()(model) # Doesn't work
    # model = Reshape((np.prod(model.shape[1:]),))(model)
    # model2 = Reshape((np.prod(model2.shape[1:]),))(model2)
    # model = Dense(n_outputs, activation='sigmoid', name='fc')(model)
    # model = Dense(n_outputs, activation=custom_activation, name='fc')(model)

    model = Flatten()(model)
    model2 = Flatten()(model2)
    fc1 = Dense(4096, activation='relu', name='fc1_1')(model)
    fc1 = Dense(4096, activation='relu', name='fc1_2')(model)
    fc1 = Dense(grid_size, activation='sigmoid', name='QRS_prediction')(model)
    fc2 = Dense(4096, activation='relu', name='fc2_1')(model2)
    fc2 = Dense(4096, activation='relu', name='fc2_2')(model2)
    fc2 = Dense(grid_size, activation='sigmoid', name='QRS_location')(model2) # activation=None ? activation='sigmoid' ?
    # final2 = tf.where(fc1 < 0.5, 0.0, fc2) # Replace QRS_location predictions with 0 if QRS_prediction = 0

    # Build model according to the above layout. This creates the Keras model instance.
    if n_outputs == 1:
        '''
        *** NOTE ***
        Doesn't work yet.
        '''
        model = Concatenate()([fc1, fc2])
        model = Model(inputs=model_input, outputs=model, name='CNN')
    else:
        model = Model(inputs=model_input, outputs=[fc1, fc2], name='CNN')
        # model = Model(inputs = [model_input, label_qrs, label_loc], outputs = [fc1, fc2], name='CNN') # if using mixed_loss

    # # Custom loss layer
    # prediction_loss = binary_crossentropy(fc1, label_qrs)
    # location_loss   = tf.where(label_qrs==1, mse(fc2, label_loc), 0)
    # mixed_loss = tf.add(loss_weights[0]*prediction_loss, loss_weights[1]*location_loss)
    # model.add_loss(mixed_loss)

    return model


'''
custom_activation
'''
def custom_activation(activation_input):

    # activation_input[0] = 0 if no QRS, 1 if QRS
    # activation_input[1] = index of QRS center, relative to segment start

    out1 = K.sigmoid(activation_input[:,0]) # Sigmoid of first column
    out2 = tf.math.round(activation_input[:,1])  # Round second column to nearest int

    return K.concatenate(out1, out2)


def custom_loss1(y_true, y_pred):
    class_penalty = 100
    class_error   = tf.subtract(y_true[:,0], y_pred[:,0])

    calc_loc_error = lambda: tf.keras.losses.mean_absolute_error(y_true[:,1], y_pred[:,1])
    loc_error = tf.where(y_true[:,0], calc_loc_error(), 0)

    return class_error*class_penalty + loc_error


'''
*** NOTE ***
This function works ONLY if the model has ONE output containing TWO elements (qrs_pred and loc_pred)
If qrs_pred_error = 1 (QRS prediction incorrect; FP or FN) : error = qrs_pred_error * qrs_pred_penalty (~= qrs_pred_penalty)
If qrs_pred_error = 0 (QRS prediction is correct):           error = qrs_pred_error * qrs_pred_penalty (~= 0) plus:
      If true_qrs = 0 (there is no QRS peak; TN):                    0
      If true_qrs = 1 (there is a QRS peak; TP) :                    mse(true_peak_loc, pred_peak_loc)
'''
def custom_loss(y_true, y_pred):
    true_qrs  = y_true[...,0]
    pred_qrs  = y_pred[...,0]
    true_loc  = y_true[...,1]
    pred_loc  = y_pred[...,1]

    qrs_pred_penalty = 10
    qrs_pred_error   = tf.abs(tf.subtract(true_qrs, pred_qrs))

    calc_loc_error = lambda: tf.keras.losses.mean_absolute_error(true_loc, pred_loc)
    loc_error = tf.where(true_qrs == 1, calc_loc_error(), 0)

    return qrs_pred_error*qrs_pred_penalty + loc_error



'''
Custom quality metric for QRS location estimates
'''
def ms_accuracy(y_true, y_pred):

    # n_ms = 5 # number of milliseconds "cushion" around target to give to count as hit (1) or miss (0)
    n_ms = 0.1 # n_ms*segment_size = number of milliseconds "cusion" around target

    y_pred = tf.math.round(y_pred) # round predictions to nearest integer (sample)
    # y_pred = tf.where(y_true == 0, 0.0, y_pred) # Set y_pred to 0 if y_true=0 (i.e., consider it a hit)
    acc = K.sum(tf.where(abs(y_true-y_pred) > n_ms, 0, 1)) / tf.size(y_true)

    return acc

'''
Custom quality metric for QRS location estimates
'''
def ms_accuracy2(y_true, y_pred):
    true_qrs  = y_true[...,0]
    true_loc  = y_true[...,1]
    pred_loc  = y_pred[...,1]

    n_ms = 5 # number of milliseconds "cushion" around target to give to count as hit (1) or miss (0)

    pred_loc = tf.where(true_qrs == 0, 0.0, pred_loc) # Set pred_loc to 0 if true_qrs=0 (i.e., consider it a hit)
    acc = K.sum(tf.where(abs(true_loc-pred_loc) > n_ms, 0, 1)) / tf.size(true_loc)

    return acc


'''
Custom quality metric for QRS location estimates
'''
def ms3_accuracy(y_true, y_pred):

    y_pred = tf.math.round(y_pred) # round predictions to nearest integer (sample)
    acc = K.sum(tf.where(abs(y_true-y_pred) > 3, 0, 1)) / tf.size(y_true)

    return acc


def validate(model, x, y, segment_size: int, sequential: bool, show_plots: bool = False, grid_size: int = 1):

    nsamples   = len(x)
    n_segments = int(nsamples/(segment_size*grid_size)) # = floor(nsamples/segment_size)

    # Split data into segments of length segment_size
    # Resulting x and y matrices are 2D tensors of size (n_segments, segment_size)
    # Each row and column of segments_y is the label corresponding to the same row and column of data in segments_x
    segments_x = np.reshape(x[0:n_segments*segment_size*grid_size], (n_segments, segment_size*grid_size, 1))
    segments_y = np.reshape(y[0:n_segments*segment_size*grid_size], (n_segments, segment_size*grid_size, 1))
    # segments_x, segments_y, n_segments = segmentize(x, y, segment_size*grid_size)

    predictions, locations = model.predict(x=segments_x)

    # Convert to 1D time series
    predictions = np.reshape(predictions, (n_segments*grid_size, 1))
    locations   = np.reshape(locations,   (n_segments*grid_size, 1))

    #
    predictions = np.round(predictions).astype(int) # 1 = QRS, 0 = no QRS
    # locations   = np.round(locations).astype(int)   # round to nearest sample

    # This chunks reverts indices in 'locations' estimates back to absolute indices, and fills non-QRS samples with None
    # y_seq = np.full_like(x, None) # make array of zeros of length len(x)
    locs = []
    vals = []
    for n, s in enumerate(predictions):
        if predictions[n]:
            idx = int(segment_size*locations[n]) + segment_size*n
            # y_seq[idx] = x[idx] # change values back to absolute indices
            locs.append(idx)
            try:
                vals.append(x[idx])
            except IndexError:
                print('Out of bounds prediction at location %d. Signal bounds: [%d, %d]' %(idx, 0, len(x)))

    if show_plots:
        # Plot
        plt.figure()
        plt.plot(x, alpha=0.5)
        plt.scatter(locs, vals, c='r', alpha=0.5)

    return locs


def peak_finder(locs, x, y, Fs, cushion=5, show_plots=True):

    TP  = np.full_like(locs, -1, dtype=np.int32)
    FP = []
    vals_TP = np.full_like(locs, -1, dtype=np.float16)
    vals_FP = []

    n_valid = 0
    for n, loc in enumerate(locs):

        tmp = x[int(max(0, loc-Fs/2)) : int(min(loc+Fs/2, len(x)))] # tmp = 1-sec window around loc
        try:
            min_height = -min(tmp) - 0.2*(max(tmp)-min(tmp)) # search only for peaks in the bottom 20% of values in 1-sec window
        except ValueError:
            print('Something is weird at index %d. int(max(0,loc-Fs/2)) = %d, int(min(loc+Fs/2, len(x))) = %d' %(loc, int(max(0, loc-Fs/2)), int(min(loc+Fs/2, len(x)))))
            continue

        search_min = int(max(0, loc-cushion))
        search_max = int(min(loc+cushion+1, len(x)))

        peak_idx, etc = sig.find_peaks(-x[search_min:search_max], height=min_height)
        idx = search_min + peak_idx
        if len(peak_idx) == 1:
            # locs[n] = idx # replace estimate with corrected estimate

            if idx < 0 or idx > len(x):
                print('Something is weird at index %d. TP idx = %d' %(loc, idx))

            acc = np.min(abs(idx - y))
            if abs(acc) <= 1:
                TP[n_valid] = idx
                vals_TP[n_valid] = x[idx]
                n_valid += 1
            else:
                TP = np.delete(TP, n_valid)
                vals_TP = np.delete(vals_TP, n_valid)
                FP.append(loc)
                vals_FP.append(x[loc])
        else:
            TP = np.delete(TP, n_valid)
            vals_TP = np.delete(vals_TP, n_valid)
            FP.append(loc)
            vals_FP.append(x[loc])

    if show_plots:
        # Plot
        plt.figure()
        plt.plot(x, alpha=0.5)
        plt.scatter(y, x[y], c='black', alpha=0.1)
        plt.scatter(TP, vals_TP, c='green', alpha=0.5)
        plt.scatter(FP, vals_FP, c='red', alpha=0.5)
        plt.title('Validation: predicted heartbeats')
        plt.ylabel('Normalized signal amplitude')
        plt.xlabel('Sample')
        plt.legend(['Signal', 'True QRS locations', 'Accurate predictions', 'Inaccurate predictions'], loc='upper right')

    return TP, FP


def plot_history(history):

    # Plot training & validation accuracy values
    keys = np.array(list(history.history.keys()))[[8,9,3,4]]

    # Plot QRS prediction accuracy
    plt.plot(history.history[keys[0]])
    plt.plot(history.history[keys[2]])
    plt.title('QRS prediction accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Prediction Train', 'Prediction Test'], loc='lower right')

    # Plot QRS location accuracy/mse
    plt.plot(history.history[keys[1]])
    plt.plot(history.history[keys[3]])
    plt.title('QRS location accuracy')
    plt.ylabel('Mean squared error')
    plt.xlabel('Epoch')
    plt.legend(['Location Train', 'Location Test'], loc='lower right')

    # plt.plot(history.history['QRS_prediction_accuracy'])
    # plt.plot(history.history['val_QRS_prediction_accuracy'])
    # plt.plot(history.history['QRS_location_mse'])
    # plt.plot(history.history['val_QRS_location_mse'])
    # # plt.plot(history.history['QRS_location_ms_accuracy'])
    # # plt.plot(history.history['val_QRS_location_ms_accuracy'])


'''
MAIN PIPELINE

1. Get data
2. Prepare data
 - ??? Preprocess data ???
 - ??? Augment data ???
 - Split into Train and Test sets
3. Model
 - Build model
 - Compile model
 - Train model
 - Test model
'''
if __name__ == '__main__':

    # Parse optional arguments:
    nargs = len(sys.argv)-1
    # argv[0] = CNNPipeline.py
    if nargs > 1:
        case = int(sys.argv[1])
    else:
        case = 3

    sequential = False
    localize   = True

    # Get data
    data, labels, validation_data, validation_labels, Fs, val_src_x, val_src_y = get_data(case, sequential)

    # Preprocess data
    seconds_per_segment = 0.5;
    segment_size        = int(Fs*seconds_per_segment) # = floor(Fs*seconds_per_segment)
    data, labels = preprocess(data, labels, segment_size, sequential)

    # Augment data
    data, labels = augment(data, labels, segment_size, reverse=False, shift=False, shift_fraction=5)

    # Split into Test and Train sets
    grid_size = 6
    Xtrain, Ytrain, Xtest, Ytest = train_test_split(data, labels, segment_size, sequential, localize=localize, grid_size=grid_size)

    # Build model
    input_len    = len(Xtrain[0])
    n_layers     = 4
    kernel_size  = [49, 25, 9, 9]
    n_filters    = [96, 128, 256, 512]
    # n_filters    = [256, 512, 1024, 2048]
    pool_size    = [2, 2, 2, 2]
    stride       = [1, 1, 1, 1]
    n_outputs    = 2 #len(Ytrain[0])
    # loss_weights = [1, 1]
    model = make_model(input_len, n_outputs, n_layers, kernel_size, n_filters, stride, pool_size, grid_size)

    # Print model summary
    model.summary()

    # Compile model
    # optimizer       = 'sgd' # 'adam'
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=0.05)
    optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=0.05)
    if type(Ytrain) == dict:
        losses          = {"QRS_prediction": "binary_crossentropy",
                           "QRS_location":   "mean_squared_error"} # custom_loss # "binary_crossentropy"
        loss_weights    = {"QRS_prediction": 2,
                           "QRS_location":   1}
        quality_metrics = {"QRS_prediction": "accuracy",
                           "QRS_location":   "mse"} #ms_accuracy}
    else:
        losses          = "binary_crossentropy"
        loss_weights    = 1
        quality_metrics = ["accuracy"]
    if n_outputs == 1:
        model.compile(optimizer=optimizer, loss=custom_loss, metrics=[binary_crossentropy, ms_accuracy2])
    elif n_outputs == 2:
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=quality_metrics)
    # model.compile(optimizer=optimizer, metrics=quality_metrics)

    # model.metrics_names.append("ms_accuracy")

    # Train model
    print("\nTRAIN SET\n=========")
    n_epochs   = 10
    batch_size = 32
    history = model.fit(x=Xtrain, y=Ytrain, validation_data=(Xtest, Ytest), epochs=n_epochs, batch_size=batch_size)

    print('Validation src x: %s' %val_src_x)
    print('Validation src y: %s' %val_src_y)
    model.save('model_CAP-all_YOLO-6-grid_aug-5.h5')

    # Plot training history
    plot_history(history)

    # Test model
    print("\nTEST SET\n========")
    # preds = model.predict(x=Xtest)
    score = model.evaluate(x=Xtest, y=Ytest)
    # print ("Loss = %0.3f" %score[0])
    # print ("Test Accuracy = %0.2f%%" %(score[1]*100))
    # QRS_prediction = np.reshape(Ytest[:,0], (len(Ytest), 1))
    # QRS_location   = np.reshape(Ytest[:,1], (len(Ytest), 1))
    # score = model.evaluate(x=Xtest, y=[QRS_prediction, QRS_location])
    print("Prediction accuracy: %0.2f%%" %(score[3]*100))
    # print("Location accuracy:   %0.2f%%" %(score[4]*100))
    print("Location mse:        %0.3f" %(score[4]))
    # print("Average distance from QRS: %0.2f" %score[4])

    # SAVE MODEL
    # model.save('tmp_model.h5')

    # Validate
    for n, ignored in enumerate(validation_data):

        # Preprocess validation set
        if not sequential:
            x, y = preprocess(validation_data[n], validation_labels[n], segment_size, sequential)
            x = x[0]
            y = y[0]

        locs = validate(model, x, y, segment_size, sequential, show_plots=True)
        TP, FP = peak_finder(locs, x, validation_labels[n], Fs, show_plots=True)

        print("After location correction algorithm, %0.2f%% of identified peaks are accurate." %(100*(len(TP)/(len(TP)+len(FP)))))
    # locs = validate(model, validation_data, validation_labels, segment_size, sequential)
    # TP, FP = peak_finder(locs, validation_data, validation_labels, Fs)

    plt.show()
    print('Done!')
