


#imports
import os
import math
import torch
import argparse
import imageio
from matplotlib import pyplot as plt
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
import cv2
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask
import numpy as np
from validation.test_parser import define_model_parser



# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
#torch imoport multiscale ssim
# from torchmetrics import multiscale ssim
import torch
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import UniversalImageQualityIndex
from torchmetrics import MeanSquaredError

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

#import multi scale ssim
# from skimage.metrics import multiscale_ssim as ms_ssim

#import tabulate
from tabulate import tabulate


# MODEL_NAME = 'PWCNet'   #'PDCNet_plus'

# PRE_TRAINED_MODEL = 'chairs_things_ft_sintel'

# ## parameters for PDCNet_plus on megadepth
# MODEL_NAME = 'PDCNet_plus'
# PRE_TRAINED_MODEL = 'megadepth'

## parameters for PWCNet on chairs_things_ft_sintel
MODEL_NAME = 'PWCNet'
PRE_TRAINED_MODEL = 'chairs_things_ft_sintel'
OPTIM_ITER = 3
LOCAL_OPTIM_ITER = 16
FLIPPING_CONDITION = False

## parameters for PWCNet_GOCor on chairs_things_ft_sintel
# MODEL_NAME = 'PWCNet_GOCor'
# PRE_TRAINED_MODEL = 'chairs_things_ft_sintel'



#we have a folder utility_images where some reference images are stored like middlebury color map guide
PATH_TO_COLOR_MAP_GUIDE = 'icons/index.jpeg'


torch.cuda.empty_cache()
# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device



## imports over--------------------------------------------------------------------------------------------------------------------------------
##function definitions starts here----------------------------------------------------------------------------------------------------------------------------

#function to make the two frames of same size

def pad_to_same_shape(im1, im2):
    # pad to same shape both images with zero
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)

    return im1, im2
#function to return image read given path to image
def read_image(path_to_image):
    try:
        #if image is greyscale, then alos we want 1x3xHxW size tensor, we will load using cv2
        if len(cv2.imread(path_to_image).shape) == 2:
            image = cv2.imread(path_to_image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(path_to_image)

        image = image[:, :, ::- 1]

    except:
        raise ValueError('It seems that the path for the image you provided does not work ! ')
    return image

#write a function to read stores frames from a file
#we will have path of pairs of frames, load both frames and return them
def read_frames(path_to_image_1, path_to_image_2):
    
    #read the two frames
    image_1 = read_image(path_to_image_1)
    image_2 = read_image(path_to_image_2)
    #save the first image shape
    target_image_shape = image_2.shape[:2]
    #pad the two frames to same shape
    image_1, image_2 = pad_to_same_shape(image_1, image_2)
    # #convert the frames to tensor
    # image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0)
    # image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0)
    #return the two frames
    return image_1, image_2, target_image_shape
#define a function to visualize the flow anf the corresponding images


def flow_visualization(estimated_flow_numpy, image_1_numpy, image_2_numpy):
    #visualize the flow
    #we will plot the flow image as per th imported function flow_to_image
    #it returns an image that is visualizing the flow
    flow_image = flow_to_image(estimated_flow_numpy)


    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(image_1_numpy)
    ax[0].set_title('First frame')
    ax[1].imshow(image_2_numpy)
    ax[1].set_title('Second frame')
    ax[2].imshow(flow_image)
    ax[2].set_title('Estimated flow')
    plt.show()

#function to plot images 
#we would be given a list of images
#and a list of titles
#we will create a figure with number of rows and columns
#and plot the images in the figure
def plot_images(images, titles ,rows =None ):
    #we will assume that the number of images is equal to the number of titles
    #but check if the number of images is equal to the number of titles
    if len(images) != len(titles):
        raise ValueError('The number of images and titles should be equal !')
    #we will check if parameter rows is apssed or not
    #if not passed we then assume 1 row and number of columns equal to number of images
    if rows is None:
        rows = 1
        columns = len(images)
    else:
        #else we will have 5 images in each row
        columns = 5
        #calculate rows
        #rows is ceil of number of images divided by 5
        rows = math.ceil(len(images)/columns)
    #create a figure with all the images as subplots
    fig, ax = plt.subplots(rows, columns, figsize=(20, 20))
    #we will iterate over all the images
    for i in range(len(images)):
        #get the row and column number
        row = i//columns
        column = i%columns
        #plot the image
        ax[i].imshow(images[i])
        #set the title
        ax[i].set_title(titles[i])
    plt.show()
    


    
#function to return SSIM, MS SSIM , LPIPS , PSNR 
#  LPIPS :A low -> perceptual similar.
# PSNR : High -> perceptual similar.
# SSIM : High -> perceptual similar.
# MS SSIM : High -> perceptual similar.
# which is a value between -1 and +1. A value of +1 indicates that the 2 given images are very similar or the same while a value of -1 indicates the 2 given images are very different.
#on the pair of images
def get_metrics(image_1, image_2):
    #we will convert the images to tensor
    image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0)
    image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0)
    #we will convert the images to float
    image_1 = image_1.float()
    image_2 = image_2.float()
    #we will convert the images to cuda
    image_1 = image_1.to(device)
    image_2 = image_2.to(device)
    #if the range of the images is 0 to 255, then we will convert it to 0 to 1
    if image_1.max() > 1:
        image_1 = image_1/255
    if image_2.max() > 1:
        image_2 = image_2/255
    # #we will convert the images to range 0 to 1
    # image_1 = image_1/255
    # image_2 = image_2/255
    #we will calculate the SSIM using the library imported sklearn
    # ssim_val = ssim(image_1, image_2, data_range=1, multichannel=True)
    #we will calculate the MS SSIM using the library imported pytorch_msssim

    #first compute the SSIM
    #create an instance of StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    #compute the SSIM
    ssim_val = ssim(image_1, image_2)
    #compute the MS SSIM
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim_val = ms_ssim(image_1, image_2)
    #we will calculate the LPIPS using the library imported pytorch_msssim
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg',weights='VGG16_Weights.DEFAULT').to(device)
    lpips_val = lpips(image_1, image_2)

    #psnr
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_val = psnr(image_1, image_2)

    #we will calculate Universal Image Quality Index
    # uqi = UniversalImageQualityIndex(data_range=1.0).to(device)
    # uqi_val = uqi(image_1, image_2)

    #we will also calculate MSE
    mse = MeanSquaredError(data_range=1.0).to(device)
    mse_val = mse(image_1, image_2)
    

    #convert all tensors to numpy array in cpu and return
    #use .detach().cpu().numpy()
    return ssim_val.detach().cpu().numpy(), ms_ssim_val.detach().cpu().numpy(), lpips_val.detach().cpu().numpy(), psnr_val.detach().cpu().numpy(),  mse_val.detach().cpu().numpy()


    #we return the SSIM, MS SSIM , LPIPS , PSNR
    # return ssim_val, ms_ssim_val, lpips_val, psnr_val

    