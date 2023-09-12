# Code for stereo matching with HMM Belief Propagation
# By Zheng Chen
# Code is based on the BP code provided in Module 9.10 and the code of naive stereo matching by Andrew Corum

import argparse
from numba import njit
import numpy as np
import os
from PIL import Image, ImageOps
import sys

def get_args():
    '''
    Get args used throughout this script
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("imageNum", type=int, choices=[1, 2, 3, 4],
        help="ID/number of directory where stereo images are located")
    parser.add_argument("-N", type=int, choices=list(range(1,10)),
        default=1, required=False,
        help="N, size of neighborhood in SSD calc, with (2N+1)x(2N+1) neighborhood")
    parser.add_argument("-M", type=str, default='linear', required=False, help='Specify the distance model class')
    return parser.parse_args()

def get_images(imNum):
    '''
    Load left and right images from the imNum directory.
    Assuming the gt, rgb_left, and rgb_right images are formatted as:
        *_left.png
        *_right.png
        *_gt.png
    Any other images will be ignored.
    '''
    files = os.listdir('./{}/'.format(imNum))
    imLeft, imRight, imTruth = None, None, None
    for f in files:
        if f.endswith('_left.png'):
            imLeft = ImageOps.grayscale(Image.open('./{}/{}'.format(imNum, f)))
        if f.endswith('_right.png'):
            imRight = ImageOps.grayscale(Image.open('./{}/{}'.format(imNum, f)))
        if f.endswith('_gt.png'):
            imTruth = ImageOps.grayscale(Image.open('./{}/{}'.format(imNum, f)))
    return imLeft, imRight, imTruth

@njit()
def SSD(NL, NR):
    '''
    Compute sum squared differences between pixel intensity values
    within the provided neighborhoods.
    '''
    h,w = NL.shape # Neighborhood size
    ssd = 0
    for y in range(h):
        for x in range(w):
            ssd += (NL[y,x] - NR[y,x])**2
    return ssd

@njit()
def basic_matching(imL, imR, N=1):
    ''' 
    Perform basic SSD matching between pixels in each scanline,
    using an (2N+1) by (2N+1) neighborhood to compute SSD.
    '''
    h,w = imL.shape # Image size
    disp = np.zeros((h,w)) # Initialize disparity matrix
    # For each scanline...
    for y in range(N, h-N):
        if y%50 == 0: print(y, '/', h-N)
        # For each pixel in scanline
        for x in range(N, w-N):
            # Max possible disparity is (2*N+1)**2
            minDisparity, minSSD = 0, 2*(2*N+1)**2 
            # For each possible disparity value given (x,y)
            for d in range(w-x):
                ssd = SSD(imL[y-N:y+N,x+d-N:x+d+N], imR[y-N:y+N,x-N:x+N])
                if ssd < minSSD: minDisparity, minSSD = d, ssd
            disp[y,x] = minDisparity
    return disp

def compute_disparity(imLeft, imRight, N=1):
    '''
    Compute the disparity matrix given a left and right image of a scene
    '''
    imL, imR = np.array(imLeft)/255.0, np.array(imRight)/255.0
    disp = basic_matching(imL, imR, N)
    return disp

def disp_to_image(disp):
    '''
    Convert the disparity map to an image.
    Also return a color-coded version.
    '''
    h, w = disp.shape
    imDisp, imColorDisp = None, np.zeros((h,w,3))
    for y in range(h):
        for x in range(w):
            red, green, blue = 0, disp[y,x], 0
            if 0 <= disp[y,x] < 128: blue = 256-2*disp[y,x]
            else: red = 2*(disp[y,x]-128)
            imColorDisp[y,x] = [int(red), int(green), int(blue)]
    imColorDisp = Image.fromarray(imColorDisp.astype(np.uint8))
    imDisp = Image.fromarray(disp.astype(np.uint8))
    return imDisp, imColorDisp

@njit()
def end_point_error(disp, truth):
    '''
    Evaluate the predicted disparity against the ground truth,
    using end point error.
    Inputs should be np.array
    '''
    h, w = disp.shape 
    epe = 0
    for y in range(h):
        for x in range(w):
            epe += abs(disp[y,x] - truth[y,x])
    return epe / (h*w)

@njit()
def error_rate(disp, truth):
    '''
    Evaluate the predicted disparity against the ground truth,
    using error rate.
    Inputs should be np.array
    '''
    h, w = disp.shape
    err = 0
    for y in range(h):
        for x in range(w):
            err += 1 if abs(disp[y,x] - truth[y,x]) > 3 else 0
    return 100 * err / (h*w)

@njit
def D(observed_disp, new_disp):
    '''
    Compute the data cost.
    '''
    return np.abs(observed_disp-new_disp)

@njit
def V(my_disp, neighbor_disp, distance_model):
    '''
    Compute the pairwise cost.
    my_disp and neighbor_disp are both hidden disps.
    '''
    abs_dist = np.abs(my_disp-neighbor_disp)
    if distance_model == 'linear':
        value = abs_dist

    return value

@njit
def find_message_loss(o_j, q_j, q_j1, last_message, distance_model):
    return D(o_j, q_j) + V(q_j, q_j1, distance_model) + last_message

@njit
def perform_HMM(disp, distance_model):
    print('---Perform HMM---')
    h, w = disp.shape
    # New disparity image
    new_disp = np.zeros(disp.shape)
    # Min disp value in the given disp image
    min_disp = np.min(disp)
    # Max disp value in the given disp image
    max_disp = np.max(disp)
    print("min disp: ", min_disp)
    print("max disp: ", max_disp)
    possible_disp = range(min_disp, max_disp)
    num_possible_disp = len(possible_disp)

    # For each scanline...
    for y in range(h):
        print(y, '/', h)
        num_pixel_per_line = w
        # Maximum iteration number
        max_iter = 20
        # left_messages[i] (one row) is the message from the i+1-the pixel to the i-th pixel 
        left_messages = np.zeros((num_pixel_per_line, num_possible_disp))
        # right_messages[i] (one row) is the message from the i-1-th pixel to the i-th pixel 
        right_messages = np.zeros((num_pixel_per_line, num_possible_disp))

        for iteration in range(max_iter):
            new_left_messages = np.zeros(left_messages.shape)
            new_right_messages = np.zeros(right_messages.shape)
            
            # Compute right messages
            for i in range(0, num_pixel_per_line-1):
                # Loop over q_{j+1}
                for neighbor_dist in possible_disp:
                    min_right_message_loss = np.inf
                    temp_right_message = None
                    # Loop over q_{j}
                    for neighbor in possible_disp:
                        # Arguments for find_message_loss: o_j, q_j, q_{j+1}, m_{j-1 -> j}(q_j)
                        right_message_loss = find_message_loss(disp[y, i], neighbor, neighbor_dist, right_messages[i][neighbor], distance_model)
                        if right_message_loss < min_right_message_loss:
                            min_right_message_loss = right_message_loss
                            temp_right_message = neighbor
                    new_right_messages[i+1][neighbor_dist] = temp_right_message

            # Compute left messages
            for i in range(1, num_pixel_per_line):
                # Loop over q_{j+1}
                for neighbor_dist in possible_disp:
                    min_left_message_loss = np.inf
                    temp_left_message = None
                    # Loop over q_{j}
                    for neighbor in possible_disp:
                        # Arguments for find_message_loss: o_j, q_j, q_{j+1}, m_{j-1 -> j}(q_j)
                        left_message_loss = find_message_loss(disp[y, i], neighbor, neighbor_dist, left_messages[i][neighbor], distance_model)
                        if right_message_loss < min_left_message_loss:
                            min_left_message_loss = left_message_loss
                            temp_left_message = neighbor
                    new_right_messages[i-1][neighbor_dist] = temp_left_message

        # Obtain the optimal disparity values
        for i in range(0, num_pixel_per_line):
            min_loss = np.inf
            temp_disp = None
            for neighbor_dist in possible_disp:
                temp_loss = right_messages[i][neighbor_dist] + left_messages[i][neighbor_dist] + D(disp[y, i], neighbor_dist)
                if temp_loss < min_loss:
                    min_loss = temp_loss
                    temp_disp = neighbor_dist
            new_disp[y, i] = temp_disp

    return new_disp

if __name__ == "__main__":
    # Get inputs
    args = get_args()
    imLeft, imRight, imTruth = get_images(args.imageNum)

    # Compute disparity
    disp = compute_disparity(imLeft, imRight, args.N)

    # Choose the distance model
    distance_model = args.M
    # Compute disparity after applying HMM (Belief Propagation)
    disp_hmm = perform_HMM(disp, distance_model)

    # Save/show results
    imDisp, imColorDisp = disp_to_image(disp)
    imDisp.show()
    imColorDisp.show()
    imDisp.save("./{0}/imDisp_{1}x{1}.png".format(args.imageNum, 2*args.N+1))
    imColorDisp.save("./{0}/imColorDisp_{1}x{1}.png".format(args.imageNum, 2*args.N+1))

    # Todo: save/show HMM results
    imHmmDiso, imColorHmmDisp = disp_to_image(disp_hmm)
    imHmmDiso.show()
    imColorHmmDisp.show()

    # Evaluate results
    disp = np.array(imDisp, dtype=np.int32)
    truth = np.array(imTruth, dtype=np.int32)
    epe = end_point_error(disp, truth)
    err = error_rate(disp, truth)
    print("End point error: {}".format(epe))
    print("Error rate: {}".format(err))

    # Todo: evaluate HMM results
