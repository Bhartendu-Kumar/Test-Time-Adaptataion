import numpy as np
import scipy.ndimage


def grad_cal(img0, img2):

    img0 = img0/ 255
    img2 = img2 / 255

    #kernels
    kernel_x = np.array([[-1,1],[-1, 1]]) / 4
    kernel_y = np.array([[-1, -1], [1, 1]]) / 4
    kernel_t = np.array([[1, 1], [1, 1]]) / 4

    ## Calculating gradients by convolving kernels
    fx = scipy.ndimage.convolve(input=img0, weights=kernel_x)+scipy.ndimage.convolve(input=img2, weights=kernel_x)
    fy = scipy.ndimage.convolve(input=img0, weights=kernel_y)+scipy.ndimage.convolve(input=img2, weights=kernel_y)
    ft = scipy.ndimage.convolve(input=img2, weights=kernel_t)+scipy.ndimage.convolve(input=img0, weights=-1*kernel_t)


    return [fx,fy,ft]


