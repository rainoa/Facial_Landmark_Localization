# Noa Raindel, June 2018

from math import floor
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import os
from utils.loadData import *

def expand_bbox(image_size,bbox,parameters):
    '''
    compute the expanded bbox
    a robust function to expand the crop image bbox even the original bbox is
    around the border of the image
    ---------------------------------------------------------------------------
    INPUT:
        image_size: a tuple   ex: (height,width)
        bbox: the ground_truth bounding boxes  ex:[x0,y0,x1,y1]
        parameters: model parameter object
    OUTPUT:
        new bbox: ex:[x0,y0,x1,y1]
    ---------------------------------------------------------------------------
    '''
    x_size,y_size = image_size
    bx0,by0,bx1,by1 = bbox
    bw = by1 - by0
    bh = bx1 - bx0
    if bw > bh:
        delta = parameters.expand_rate * bw
        if by1 + delta > y_size:
            new_by1 = y_size
        else:
            new_by1 = int(floor(by1 + delta))
        if by0 - delta < 0:
            new_by0 = 0
        else:
            new_by0 = int(floor(by0 - delta))
        new_w = new_by1 - new_by0
        delta_h = (new_w - bh) / 2
        if bx0 - delta_h < 0:
            new_bx0 = 0
        else:
            new_bx0 = int(floor(bx0 - delta_h))
        if bx1 + delta_h > x_size:
            new_bx1 = x_size
        else:
            new_bx1 = int(floor(bx1 + delta_h))
    else:
        delta = parameters.expand_rate * bh
        if bx1 + delta > x_size:
            new_bx1 = x_size
        else:
            new_bx1 = int(floor(bx1 + delta))
        if bx0 - delta < 0:
            new_bx0 = 0
        else:
            new_bx0 = int(floor(bx0 - delta))
        new_h = new_bx1 - new_bx0
        delta_w = (new_h - bw) / 2
        if by0 - delta_w < 0:
            new_by0 = 0
        else:
            new_by0 = int(floor(by0 - delta_w))
        if by1 + delta_w > y_size:
            new_by1 = y_size
        else:
            new_by1 = int(floor(by1 + delta_w))
    return new_bx0,new_by0,new_bx1,new_by1


def crop_and_resize_image(image_name,bbox,parameters, print_image_path=False):
    '''
    crop and resize the image given the ground truth bounding boxes
    also, compute the new coordinates according to transformation
    ---------------------------------------------------------------------------
    INPUT:
        image_name: a string without extension  ex: 'image_0007'
        bbox: the ground_truth bounding boxes  ex:[x0,y0,x1,y1]
        parameters: model parameter object
        print_image_path: boolean indicating if to print the image full path
    OUTPUT:
        im_resize: a numpy array of the image after crop and resize
        landmarks: new landmarks accordance with new image
    ---------------------------------------------------------------------------
    '''
    image_path = parameters.dataset + '/' + parameters.train_or_test + 'set/imgs/' + image_name
    if print_image_path:
        print(image_path)
    assert os.path.exists(image_path)
    im = Image.open(image_path)
    bbox = expand_bbox(im.size,bbox,parameters)
    im_crop = im.crop(bbox)
    im_resize = im_crop.resize(parameters.new_size)
    
    #compute the new landmarks according to transformation procedure
    landmarks = load_landmarks(image_name,parameters)
    landmarks = landmarks - (bbox[:2])
    landmarks = landmarks * im_resize.size / im_crop.size

    
    return np.array(im_resize),landmarks.astype(int)