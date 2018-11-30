# Noa Raindel, June 2018

import numpy as np
import scipy.io as sio
import os


def get_images_names(parameters):
    '''
    get a list containing all the paths of images in the trainset
    ---------------------------------------------------------------------------
    INPUT:
        parameters: model parameter object
    OUTPUT:
        a list with all the images' names, including extension
    ---------------------------------------------------------------------------
    '''
    folder_path = parameters.dataset + '/' + parameters.train_or_test + 'set/imgs'
    assert os.path.exists(folder_path)
    assert os.path.isdir(folder_path)
    imageNames = os.listdir(folder_path)
    return imageNames


def load_boxes(parameters):
    '''
   read a mat file and get a dictionary of all the bounding boxes
    ---------------------------------------------------------------------------
    INPUT:
        parameters: model parameter object
    OUTPUT:
        ret: a dictionary, keys are image names' (including extension), values are 
             the bounding boxes  ex:[x0,y0,x1,y1]
    ---------------------------------------------------------------------------
    '''
    # just some ugly translations from the very nested MATLAB representation of the bbox information
    bboxes1 = sio.loadmat(parameters.dataset + '/bounding_boxes_' +parameters.dataset 
                          + '_'+parameters.train_or_test +'set.mat')['bounding_boxes']
    ret = {}
    for bb in bboxes1[0]:
        ret[bb[0][0][0][0]] = list(bb[0][0][1][0])
    return ret


def load_landmarks(image_name,parameters):
    '''
    load the landmarks coordinates from .pts file
    ---------------------------------------------------------------------------
    INPUT:
        image_name: a string with extension   ex: 'image_0122.png'
        parameters: model parameter object
    OUTPUT:
        a numpy array containing all the points
    ---------------------------------------------------------------------------
    '''    
    file_path = parameters.dataset + '/' + parameters.train_or_test + 'set/pts/' + image_name[:-4] +'.pts'  
    assert os.path.exists(file_path)
    with open(file_path) as f: 
        rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows[rows.index('{') + 1:rows.index('}')]]
    return np.array([list([float(point) for point in coords]) for coords in coords_set])

