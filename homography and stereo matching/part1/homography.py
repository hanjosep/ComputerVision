
from sklearn.cluster import KMeans
from PIL import Image
import sys
import os
import math
import numpy as np
from numba import njit

# get lowest sum-distance from image origin to get the kmeans_clusters origin
# lowest_x0: the lowest point downward that isn't the origin
# lowest_x1: the lowest point eastward that isn't the origin
# farthest: the point with highest sum_distance from origin
# NOTE: this code fails if lower left point is closer to origin than the top left point.
# This does not occur in any of the bilboard images however.
def order_points_josep(arr):
    origin = arr[arr.sum(axis=1).argmin()] 

    lowest_x1 = arr[arr[:,1].argmin()]
    # a[arr[:,1].argmin()],a
    if np.array_equal(lowest_x1, origin):
        x1_index = np.where(arr[:,1] == np.partition(arr[:,1],1)[1])[0][0]
        lowest_x1 = arr[x1_index]
    lowest_x0 = arr[arr[:,0].argmin()]
    if np.array_equal(lowest_x0, origin):
        x0_index = np.where(arr[:,0] == np.partition(arr[:,0],1)[1])[0][0]
        lowest_x0 = arr[x0_index]

    farthest  = arr[arr.sum(axis=1).argmax()] 

    return np.array([origin,lowest_x0,lowest_x1,farthest])

# influence for this code:
# splitting between colors: https://stackoverflow.com/questions/60051941/find-the-coordinates-in-an-image-where-a-specified-colour-is-detected
# logical and: https://stackoverflow.com/questions/13869173/numpy-find-index-of-the-elements-within-range
@njit()
def get_four_points_helper(npim, color = 'Blue'):
    # find points where the Red and Green values are below 50 and Blue values are above 200. 
    if color == 'Red': 
        x,y = np.where(np.logical_and(np.logical_and(npim[:,:,0] >200, npim[:,:,1] < 50), npim[:,:,2] < 50))
    if color == 'Green':
        x,y = np.where(np.logical_and(np.logical_and(npim[:,:,0] <50, npim[:,:,1] > 200), npim[:,:,2] < 50))
    if color == 'Blue':
        x,y = np.where(np.logical_and(np.logical_and(npim[:,:,0] <50, npim[:,:,1] < 50), npim[:,:,2] > 200))
    # zip the x,y coordinates into a list of 2-d array.
    zipped = np.column_stack((x,y))
    return zipped
    
def get_four_points(npim, color = "Blue"):
    '''
    Given a preprocessed image, return four points to apply homography.

    Params: 
        npim (2d np.array): np converted color image of preprocessed image.
        color: the kind of color to search for.
        NOTE: this image is preprocessed with colored annotations through Microsoft Paint.
        NOTE: because the image is saved as a JPG, the color values are compressed, and thus
            we cannot search for exact matches for the blue value.
        NOTE: since the image is extracted using Image.open, 
            the color ordering is RGB. so npim[:,:,0] returns RED values for the given point
            and npim[:,:,1] GREEN, npim[:,:,2] BLUE.
    Returns:
        4 cluster centers found using the KMeans clustering algorithm
 ''' 
    zipped = get_four_points_helper(npim,color)

    # fit a KMeans clustering algorithm to find the 4 converging points. 
    # NOTE: this breaks the njit compiler. If we are to optimize for speed, we
    # separate the program from the pure nparray and the KMeans algorithm. 
    
    means = KMeans(4).fit(zipped)
    result = order_points_josep(means.cluster_centers_)
    # return just the 4 cluster centers found from the KMeans clustering algorithm.
    return result


def get_source_corners(npim):
    '''
    Given a source image (e.g. the stock minion image) converted into a nparray,
    return the four corners of the image in the same ordering as the corners found in 
    get_four_points. 
    '''
    # print('hi')
    # print('hi')
    npim_x0,npim_y0 = (0,0)
    npim_x1,npim_y1 = (npim.shape[0],0)
    npim_x2,npim_y2 = (0,npim.shape[1])
    npim_x3,npim_y3 = (npim.shape[0],npim.shape[1])

    result = np.array([
        [ npim_x0,npim_y0],
        [ npim_x2,npim_y2],
        [ npim_x1,npim_y1],
        [ npim_x3,npim_y3],
        
    ])
    return result

# Given four corners of source and the corresponding four points of the target image,
# return the transformation matrix that allows this transformation.
def get_matrix(source, target):

    x0,y0 = source[0]
    x1,y1 = source[1]
    x2,y2 = source[2]
    x3,y3 = source[3]

    u0,v0 = target[0]
    u1,v1 = target[1]
    u2,v2 = target[2]
    u3,v3 = target[3]

    A = np.array([
        [x0 ,y0, 1, 0, 0 , 0,-x0*u0, -y0*u0, -u0],
        [x1 ,y1, 1, 0, 0 , 0,-x1*u1, -y1*u1, -u1],

        [x2 ,y2, 1, 0, 0 , 0,-x2*u2, -y2*u2, -u2],
        [x3 ,y3, 1, 0, 0 , 0,-x3*u3, -y3*u3, -u3],

        [ 0 , 0, 0,x0, y0, 1,-x0*v0, -y0*v0, -v0],
        [ 0 , 0, 0,x1, y1, 1,-x1*v1, -y1*v1, -v1],

        [ 0 , 0, 0,x2, y2, 1,-x2*v2, -y2*v2, -v2],
        [ 0 , 0, 0,x3, y3, 1,-x3*v3, -y3*v3, -v3],

        [ 0 , 0, 0, 0,  0, 0,     0,      0,   1],
        ])

    b = np.zeros(9)
    b[8] = 1
    h = np.linalg.solve(A,b)
    h = h.reshape((3,3))
    return h

# @njit
def apply_homography(source_image, target_image, h_matrix):
    new_image = target_image
    for i in range(0,source_image.shape[0]-1):
        for j in range(0, source_image.shape[1]-1):
            bi = np.float64([i,j,1])
            # bi = bi.astype('float64')
            # print(h_matrix.dtype, bi.dtype)
            newx, newy,   w= np.matmul(h_matrix,bi)
            # projected_point = np.matmul(h_matrix,bi)
            # projected_point = h_matrix.dot(bi)
            # pp = np.int64(projected_point / projected_point[-1])
            newx = np.round(newx / w )
            newy = np.round(newy / w )
            try:
                new_image[int(newx)][int(newy)]=source_image[i][j]
                # new_image[int(pp[0])][int(pp[1])]=source_image[i][j]
                # new_image[pp[0]][pp[1]]=source_image[i][j]
            # new_target[int(newx)][int(newy)]=source_image[int(i/free )][int(j/free)]
            except:
                pass
    return new_image
import PIL
if __name__ == '__main__':

    color = 'Blue'
    if len(sys.argv) == 5:
        source_file, target_file, annot_file, output_name = sys.argv[1:5]
    if len(sys.argv) == 6:
        source_file, target_file, annot_file,color, output_name = sys.argv[1:6]




    target_image = Image.open(target_file)
    target_image = target_image.convert('RGB')
    target_image = np.array(target_image)

    source_image = Image.open(source_file)
    # print(target_image.size)
    # print(source_image.size[0]* source_image.size[1] , target_image.shape[0] * target_image.shape[1])
    if source_image.size[0] * source_image.size[1] > target_image.shape[0] * target_image.shape[1]:
        print('resizing source image (source image is larger than target)')
        source_image = source_image.resize((target_image.shape[0], target_image.shape[1]),resample = PIL.Image.BILINEAR)
    source_image = np.array(source_image)
    # source_image = source_image.resize(target_image.size)

    annot_image = Image.open(annot_file)
    annot_image = np.array(annot_image)

    source = get_source_corners(source_image)
    target = get_four_points(annot_image, color)
    H = get_matrix(source, target)
    for i,j in source:
        print(i,j)
        # projected_point = operator.imatmul(H,[i,j,1])
        projected_point = (H @[i,j,1])
        projected_point = (H.dot([i,j,1]))
        print(projected_point / projected_point[-1])
    print(target,H)
    print(target_image.shape)
    # print(target)
    # print(source,target,H)

    homography_image = apply_homography(source_image, target_image, H)
    homography_image = Image.fromarray(homography_image)
    homography_image.save(output_name)
    print('Done. Output image in', output_name)