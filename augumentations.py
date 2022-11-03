#we will use pretrained optical flow model from the script: pre_trained_model_wrappers.py
#we import the class: Args
import importlib
import pre_trained_model_wrapper_new
#reload

importlib.reload(pre_trained_model_wrapper_new)
from pre_trained_model_wrapper_new import Args      #class storing arguments
#we import functions: build_model ,display_results , return_both_flows , compare_optical_flow 
from pre_trained_model_wrapper_new import build_model, display_results, return_both_flows, compare_optical_flow, get_pretrained_model , get_optical_flow


#import model name and pretrained model name from script parameters_new.py
import parameters_new
#reload

importlib.reload(parameters_new)
from parameters_new import MODEL_NAME, PRE_TRAINED_MODEL, CURRENT_MODEL

#class Args: is used to store arguments

# build_model: is used to build the model  'PWCNet' on 'chairs_things_ft_sintel'
# compare_optical_flow: is used to compare the optical flow of two images
#get_pretrained_model: is used to get the pretrained model given model and dataset


import os
import numpy as np
from matplotlib import pyplot as plt
# from Horn_Schunk.Horn_schunk import *
# from Interpolation.interpolation import *
# from Lukas_kanade.lukas_kanade import *
# from Multiscale_Lukas_kanade.Multiscale_lukas_kanade import *
# from Multiscale_Horn_schunk.multiscale_horn_schunk import *
import cv2
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import os
import random
import warnings

# warnings.filterwarnings("ignore")

#seed for reproducibility
random.seed(123)


def warp(img, flow, grad):
    img = (img / 255) + (flow[0] * grad[0] + flow[1] * grad[1] + grad[2])
    return img

def put_rec(img):
    #define upper bound as 20
    ub = 32
    center = (random.randint(ub, img.shape[0] - ub), random.randint(ub, img.shape[1] - ub))
    size = random.randint(1, ub)
    size = (size, size)
    pos = ((center[0] - size[0], center[1] - size[1]), (center[0] + size[0], center[1] + size[1]))
    out_img = cv2.rectangle(img, pos[0], pos[1], (0, 0, 0), -1)
    return out_img,pos

def dummy_of(img):
    #generate numpy matrix of size img.shape[0] x img.shape[1] x 2 with random values between -30 and 30 of datatype float32
    return np.random.randint(-30, 30, (img.shape[0], img.shape[1], 2)).astype(np.float32)

#replace dummy_of : with get_optical_flow


#dummy function
def test0():
    print("test1")


#wew ill need to define two functions: one for corridor dataset and one fro sphere dataset: exactly doing what the below code is doing as we will import this file in the main file

def create_augmented_corridor(num_augumentations = 100, corridor_path = 'Assignment_datasets/corridor/' , save_dir = 'data/corridor/' , model_type = CURRENT_MODEL,model = MODEL_NAME, pre_trained_model = PRE_TRAINED_MODEL ):
    #coctenatesave_dir and MODEL_NAME to create a path to save the augmented data
    #if model name is empty then do not end the string with '/' but if not then end it with '/'
    if model_type == '':
        save_dir = save_dir
    else:
        save_dir = save_dir + model_type + '/'
    for i in range(0, 9, 1):
        print("index", i, "starting")
        cor = corridor_path
        frame1 = str("bt.00") + str(i) + str(".pgm")
        if i + 1 != 10:
            frame2 = str("bt.00") + str(i + 1) + str(".pgm")
        else:
            frame2 = str("bt.0") + str(i + 1) + str(".pgm")

        frame1_path = str(cor) + str(frame1)
        frame2_path = str(cor) + str(frame2)

        #create directory for storing results if not present
        if not os.path.exists(str(save_dir)+ str(i)+str("_")+str(i+1)):
            os.makedirs(str(save_dir)+ str(i)+str("_")+str(i+1))

        #create directory if not present with name as frame_i inside Augmented_i_i+1
        if not os.path.exists(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i)):
            os.makedirs(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i))

        #create directory if not present with name as frame i+1 inside Augmented_i_i+1
        if not os.path.exists(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + str(i+1)):
            os.makedirs(str(save_dir)+  str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1))


        frame1_img = cv2.imread(frame1_path)
        frame2_img = cv2.imread(frame2_path)


        frame1_img = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
        frame2_img = cv2.cvtColor(frame2_img, cv2.COLOR_BGR2GRAY)

        #create directory named as 0 inside Augmented_i_i+1/Frame_i
        if not os.path.exists(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(0)):
            os.makedirs(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(0))
        #store frame1_img in Augmented_i_i+1/Frame_i/0 as 0.pgm
        cv2.imwrite(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(0) + "/" + str(0) + ".pgm", frame1_img)

        #create directory named as 0 inside Augmented_i_i+1/Frame_i+1
        if not os.path.exists(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(0)):
            os.makedirs(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(0))
        #store frame2_img in Augmented_i_i+1/Frame_i+1/0 as 0.pgm
        cv2.imwrite(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(0) + "/" + str(0) + ".pgm", frame2_img)


        # OF = dummy_of(frame1_img , frame2_img)
        #replace dummy_of with get_optical_flow: which takes arguments: model = MODEL_NAME, pre_trained_model = PRE_TRAINED_MODEL, plot= False, img_pth_1 , img_pth_2 
        OF = get_optical_flow(model = model , pre_trained_model = pre_trained_model, plot= False, img_pth_1 = frame1_path, img_pth_2 = frame2_path)
        #loop for generating 100 augmented images

        for j in range(1, num_augumentations, 1):
            # copy frame1_img and frame2_img to frame1_img_copy and frame2_img_copy
            frame1_img_copy = frame1_img.copy()
            frame2_img_copy = frame2_img.copy()

            frame1_img_copy,pos = put_rec(frame1_img_copy)
            # declare a black pixel of size 1
            black_pixel = 0
            # loop over the pixels of the rectangle
            for m in range(pos[0][0],pos[1][0]):
                for n in range(pos[0][1],pos[1][1]):
                    #round up the OF float values to int
                    x = int(round(OF[m][n][0]))
                    y = int(round(OF[m][n][1]))
                    new_i = m + x
                    new_j = n + y
                    #if new_i and new_j are within the image
                    if new_i >= 0 and new_i < frame1_img_copy.shape[0] and new_j >= 0 and new_j < frame1_img_copy.shape[1]:
                        #typecase new_i and new_j to int
                        new_i = int(new_i)
                        new_j = int(new_j)
                        #replace the pixel at new_i,new_j of frame2_img_copy with black pixel
                        frame2_img_copy[new_i][new_j] = 0

            #create directory named as j inside Augmented_i_i+1/Frame_i
            if not os.path.exists(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(j)):
                os.makedirs(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(j))
            #store frame1_img_copy in Augmented_i_i+1/Frame_i/j as j.pgm
            cv2.imwrite(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(j) + "/" + str(j) + ".pgm", frame1_img_copy)

            #create directory named as j inside Augmented_i_i+1/Frame_i+1
            if not os.path.exists(str(save_dir) + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(j)):
                os.makedirs(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(j))
            #store frame2_img_copy in Augmented_i_i+1/Frame_i+1/j as j.pgm
            cv2.imwrite(str(save_dir)+ str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(j) + "/" + str(j) + ".pgm", frame2_img_copy)



#similarly create for sphere
def create_augmented_sphere(sphere_path = 'Assignment_datasets/sphere/', save_dir = 'data/sphere/', model_type = CURRENT_MODEL,model = MODEL_NAME, pre_trained_model = PRE_TRAINED_MODEL ):
    # Sphere Dataset
    for i in range(0, 17, 1):
        print("index", i, "starting")
        sph = sphere_path
        frame1 = str("sphere.") + str(i) + str(".ppm")
        frame2 = str("sphere.") + str(i + 1) + str(".ppm")


        frame1_path = str(sph) + str(frame1)
        frame2_path = str(sph) + str(frame2)

        frame1_img = cv2.imread(frame1_path)
        frame2_img = cv2.imread(frame2_path)


        frame1_img = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
        frame2_img = cv2.cvtColor(frame2_img, cv2.COLOR_BGR2GRAY)

        #create directory augmented_sphere_dataset_i if not present
        os.makedirs(str(save_dir)+"Augmented_" +str(i),exist_ok=True)

        #create the directory as frame_i inside augmented_sphere_dataset_i
        os.makedirs(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i),exist_ok=True)

        #create the directory as frame_i+1 inside augmented_sphere_dataset_i
        os.makedirs(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i+1),exist_ok=True)

        #create the directory named as 0 inside augmented_sphere_dataset_i/frame_i
        os.makedirs(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i)+"/0",exist_ok=True)

        #create the directory named as 0 inside augmented_sphere_dataset_i/frame_i+1
        os.makedirs(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i+1)+"/0",exist_ok=True)

        #store the frame1_img in augmented_sphere_dataset_i/frame_i/0/ as 0.pgm
        cv2.imwrite(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i)+"/0/0.ppm",frame1_img)

        #store the frame2_img in augmented_sphere_dataset_i/frame_i+1/0/ as 0.pgm
        cv2.imwrite(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i+1)+"/0/0.ppm",frame2_img)

        OF = get_optical_flow(model = model , pre_trained_model = pre_trained_model, plot= False, img_pth_1 = frame1_path, img_pth_2 = frame2_path)

        #loop for generating 10 augmented images
        for j in range(1,11,1):
            #copy the frame1_img and frame2_img to frame1_img_temp and frame2_img_temp
            frame1_img_temp = frame1_img.copy()
            frame2_img_temp = frame2_img.copy()

            frame1_img_temp,pos = put_rec(frame1_img_temp)
            black_pixel = 0

            #loop over the values of pos
            for m in range(pos[0][0],pos[1][0]):
                for n in range(pos[0][1],pos[1][1]):
                    # round up the OF float values to int
                    x = int(round(OF[m][n][0]))
                    y = int(round(OF[m][n][1]))
                    new_i = m + x
                    new_j = n + y
                    # if new_i and new_j are within the image
                    if new_i >= 0 and new_i < frame1_img_copy.shape[0] and new_j >= 0 and new_j < frame1_img_copy.shape[1]:
                        # typecase new_i and new_j to int
                        new_i = int(new_i)
                        new_j = int(new_j)
                        # replace the pixel at new_i,new_j of frame2_img_copy with black pixel
                        frame2_img_copy[new_i][new_j] = 0

            #create the directory names as j inside augmented_sphere_dataset_i/frame_i
            os.makedirs(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i)+"/"+str(j),exist_ok=True)

            #create the directory names as j inside augmented_sphere_dataset_i/frame_i+1
            os.makedirs(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i+1)+"/"+str(j),exist_ok=True)

            #store the frame1_img_temp in augmented_sphere_dataset_i/frame_i/j/ as j.ppm
            cv2.imwrite(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i)+"/"+str(j)+"/"+str(j)+".ppm",frame1_img_temp)

            #store the frame2_img_temp in augmented_sphere_dataset_i/frame_i+1/j/ as j.ppm
            cv2.imwrite(str(save_dir)+"Augmented_"+str(i)+"/frame_"+str(i+1)+"/"+str(j)+"/"+str(j)+".ppm",frame2_img_temp)

    #print message as done
    print("Completed")


if __name__ == "__main__":
    ## Hyper parameter
    lambada = 1
    max_iter = 400
    epsilon = 0.001
    result = "Sphere_augmented_dataset"
    print("Started execution...")

    # Creating dataframe for scores
    # forward_pred_scores = pd.DataFrame(
    #     columns=["horn_schunk_ssim", "horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
    #              ], index=list(range(2, 11, 1)))

    # backward_pred_scores = pd.DataFrame(
    #     columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
    #              "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
    #              "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(0, 10, 2)))
    #
    # interpolated_frame_scores = pd.DataFrame(
    #     columns=["horn_schunk_ssim", "horn_schunk_psnr", "multiscale_horn_schunk_ssim",
    #              "multiscale_horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",
    #              "multiscale_lukas_kanade_ssim", "multiscale_lukas_kanade_psnr"], index=list(range(1, 10, 2)))

    # Corridor Dataset
    for i in range(0, 9, 1):
        print("index", i, "starting")
        cor = "./Dataset/corridor/"
        frame1 = str("bt.00") + str(i) + str(".pgm")
        if i + 1 != 10:
            frame2 = str("bt.00") + str(i + 1) + str(".pgm")
        else:
            frame2 = str("bt.0") + str(i + 1) + str(".pgm")

        frame1_path = str(cor) + str(frame1)
        frame2_path = str(cor) + str(frame2)

        #create directory for storing results if not present
        if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1)):
            os.makedirs("Augmented_" + str(i)+str("_")+str(i+1))

        #create directory if not present with name as frame_i inside Augmented_i_i+1
        if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i)):
            os.makedirs("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i))

        #create directory if not present with name as frame i+1 inside Augmented_i_i+1
        if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1) + "/" + str(i+1)):
            os.makedirs("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1))


        frame1_img = cv2.imread(frame1_path)
        frame2_img = cv2.imread(frame2_path)


        frame1_img = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
        frame2_img = cv2.cvtColor(frame2_img, cv2.COLOR_BGR2GRAY)

        #create directory named as 0 inside Augmented_i_i+1/Frame_i
        if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(0)):
            os.makedirs("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(0))
        #store frame1_img in Augmented_i_i+1/Frame_i/0 as 0.pgm
        cv2.imwrite("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(0) + "/" + str(0) + ".pgm", frame1_img)

        #create directory named as 0 inside Augmented_i_i+1/Frame_i+1
        if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(0)):
            os.makedirs("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(0))
        #store frame2_img in Augmented_i_i+1/Frame_i+1/0 as 0.pgm
        cv2.imwrite("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(0) + "/" + str(0) + ".pgm", frame2_img)


        OF = dummy_of(frame1_img)
        #loop for generating 100 augmented images

        for j in range(1, 100, 1):
            # copy frame1_img and frame2_img to frame1_img_copy and frame2_img_copy
            frame1_img_copy = frame1_img.copy()
            frame2_img_copy = frame2_img.copy()

            frame1_img_copy,pos = put_rec(frame1_img_copy)
            # declare a black pixel of size 1
            black_pixel = 0
            # loop over the pixels of the rectangle
            for m in range(pos[0][0],pos[1][0]):
                for n in range(pos[0][1],pos[1][1]):
                    #round up the OF float values to int
                    x = int(round(OF[m][n][0]))
                    y = int(round(OF[m][n][1]))
                    new_i = m + x
                    new_j = n + y
                    #if new_i and new_j are within the image
                    if new_i >= 0 and new_i < frame1_img_copy.shape[0] and new_j >= 0 and new_j < frame1_img_copy.shape[1]:
                        #typecase new_i and new_j to int
                        new_i = int(new_i)
                        new_j = int(new_j)
                        #replace the pixel at new_i,new_j of frame2_img_copy with black pixel
                        frame2_img_copy[new_i][new_j] = 0

            #create directory named as j inside Augmented_i_i+1/Frame_i
            if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(j)):
                os.makedirs("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(j))
            #store frame1_img_copy in Augmented_i_i+1/Frame_i/j as j.pgm
            cv2.imwrite("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i) + "/" + str(j) + "/" + str(j) + ".pgm", frame1_img_copy)

            #create directory named as j inside Augmented_i_i+1/Frame_i+1
            if not os.path.exists("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(j)):
                os.makedirs("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(j))
            #store frame2_img_copy in Augmented_i_i+1/Frame_i+1/j as j.pgm
            cv2.imwrite("Augmented_" + str(i)+str("_")+str(i+1) + "/" + "Frame" + str(i+1) + "/" + str(j) + "/" + str(j) + ".pgm", frame2_img_copy)




    #
    #     print("Calculating flow using hornshunk")
    #     # Calculating flow using hornshunk
    #     forward_flow, forward_grad = horn_schunk_flow(frame1_img, frame2_img, lambada, max_iter, epsilon)
    #     # backward_flow, backward_grad = horn_schunk_flow(frame3_gt_img, frame1_img, lambada, max_iter, epsilon)
    #
    #     print("forward prediction")
    #     # forward prediction
    #     predicted_img = warp(frame2_img, forward_flow, forward_grad)
    #     os.makedirs("./"+result+"/Horn_schunk_results/corridor/forward_prediction",exist_ok=True)
    #     temp = "./"+result+"/Horn_schunk_results/corridor/forward_prediction/HSK_forward_pred_" + str(i + 1) + ".pgm"
    #
    #     print("Writing Predicted Image")
    #     cv2.imwrite(temp, predicted_img * 255)
    #
    #     print("Calculating SSIM and PSNR")
    #     forward_pred_scores.horn_schunk_ssim[i + 2] = ssim(frame3_gt_img, predicted_img * 255)
    #     forward_pred_scores.horn_schunk_psnr[i + 2] = psnr(frame3_gt_img, predicted_img * 255)
    #
    #     # # backward prediction
    #     # img = warp(frame3_gt_img, backward_flow, backward_grad)
    #     # temp = "./Results/Horn_schunk_results/corridor/backward_prediction/backward_pred_" + str(i) + ".pgm"
    #     # cv2.imwrite(temp, img * 255)
    #     # backward_pred_scores.horn_schunk_ssim[i] = ssim(frame1_img, img * 255)
    #     # backward_pred_scores.horn_schunk_psnr[i] = psnr(frame1_img, img * 255)
    #
    #     # # Interpolated
    #     # img = interpolate1(frame1_img, frame3_gt_img, forward_flow, backward_flow)
    #     # temp = "./Results/Horn_schunk_results/corridor/interpolated_frame/interpolated_" + str(i + 1) + ".pgm"
    #     # cv2.imwrite(temp, img)
    #     # interpolated_frame_scores.horn_schunk_ssim[i + 1] = ssim(frame2_img, img)
    #     # interpolated_frame_scores.horn_schunk_psnr[i + 1] = psnr(frame2_img, img)
    #
    #     # Calculating flow using multiscale horn schunk
    #     # forward_flow, forward_grad = multiscale_horn_schunk_flow(frame1_img, frame3_gt_img, lambada, max_iter, epsilon,
    #     #                                                          4)
    #     # backward_flow, backward_grad = multiscale_horn_schunk_flow(frame3_gt_img, frame1_img, lambada, max_iter,
    #     #                                                            epsilon, 4)
    #
    #     # forward prediction
    #     # img = warp(frame1_img, forward_flow, forward_grad)
    #     # temp = "./Results/Multiscale_Horn_schunk_results/corridor/forward_prediction/forward_pred_" + str(
    #     #     i + 2) + ".pgm"
    #     # cv2.imwrite(temp, img * 255)
    #     # forward_pred_scores.multiscale_horn_schunk_ssim[i + 2] = ssim(frame3_gt_img, img * 255)
    #     # forward_pred_scores.multiscale_horn_schunk_psnr[i + 2] = psnr(frame3_gt_img, img * 255)
    #
    #     # # backward prediction
    #     # img = warp(frame3_gt_img, backward_flow, backward_grad)
    #     # temp = "./Results/Multiscale_Horn_schunk_results/corridor/backward_prediction/backward_pred_" + str(i) + ".pgm"
    #     # cv2.imwrite(temp, img * 255)
    #     # backward_pred_scores.multiscale_horn_schunk_ssim[i] = ssim(frame1_img, img * 255)
    #     # backward_pred_scores.multiscale_horn_schunk_psnr[i] = psnr(frame1_img, img * 255)
    #
    #     # Interpolated
    #     # img = interpolate1(frame1_img, frame3_gt_img, forward_flow, backward_flow)
    #     # temp = "./Results/Multiscale_Horn_schunk_results/corridor/interpolated_frame/interpolated_" + str(
    #     #     i + 1) + ".pgm"
    #     # cv2.imwrite(temp, img)
    #     # interpolated_frame_scores.multiscale_horn_schunk_ssim[i + 1] = ssim(frame2_img, img)
    #     # interpolated_frame_scores.multiscale_horn_schunk_psnr[i + 1] = psnr(frame2_img, img)
    #
    #     # Calculating flow using lukas kanade
    #
    #     print("Calculating Optical Flow using Lucas Kanade method")
    #     forward_flow, forward_grad = lukas_kanade_flow(frame1_img, frame2_img, 9)
    #     # backward_flow, backward_grad = lukas_kanade_flow(frame3_gt_img, frame1_img, 9)
    #
    #     print("Predicting Image")
    #     # forward prediction
    #     predicted_img = warp(frame2_img, forward_flow, forward_grad)
    #
    #     os.makedirs("./"+result+"/Lukas_kanade_results/corridor/forward_prediction",exist_ok=True)
    #     temp = "./"+result+"/Lukas_kanade_results/corridor/forward_prediction/LK_forward_pred_" + str(i + 1) + ".pgm"
    #
    #     print("Writing Image")
    #     cv2.imwrite(temp, predicted_img * 255)
    #
    #     print("Calculating SSIM and PSNR")
    #     forward_pred_scores.lukas_kanade_ssim[i + 2] = ssim(frame3_gt_img, predicted_img * 255)
    #     forward_pred_scores.lukas_kanade_psnr[i + 2] = psnr(frame3_gt_img, predicted_img * 255)
    #
    #     # backward prediction
    #     # img = warp(frame3_gt_img, backward_flow, backward_grad)
    #     # cv2.imwrite("./Results/Lukas_kanade_results/corridor/backward_prediction/backward_pred_" + str(i) + ".pgm",
    #     #             img * 255)
    #     # backward_pred_scores.lukas_kanade_ssim[i] = ssim(frame1_img, img * 255)
    #     # backward_pred_scores.lukas_kanade_psnr[i] = psnr(frame1_img, img * 255)
    #
    #     # Interpolated
    #     # img = interpolate1(frame1_img, frame3_gt_img, forward_flow, backward_flow)
    #     # temp = "./Results/Lukas_kanade_results/corridor/interpolated_frame/interpolated_" + str(i + 1) + ".pgm"
    #     # cv2.imwrite(temp, img)
    #     # interpolated_frame_scores.lukas_kanade_ssim[i + 1] = ssim(frame2_img, img)
    #     # interpolated_frame_scores.lukas_kanade_psnr[i + 1] = psnr(frame2_img, img)
    #
    #     # Calculating flow using multiscale Lukas kanade
    #     # forward_flow, forward_grad = multiscale_lukas_kanade_flow(frame1_img, frame3_gt_img, 9, 4)
    #     # backward_flow, backward_grad = multiscale_lukas_kanade_flow(frame3_gt_img, frame1_img, 9, 4)
    #
    #     # forward prediction
    #     # img = warp(frame1_img, forward_flow, forward_grad)
    #     # cv2.imwrite(
    #     #     "./Results/Multiscale_Lukas_kanade_results/corridor/forward_prediction/forward_pred_" + str(i + 2) + ".pgm",
    #     #     img * 255)
    #     # forward_pred_scores.multiscale_lukas_kanade_ssim[i + 2] = ssim(frame3_gt_img, img * 255)
    #     # forward_pred_scores.multiscale_lukas_kanade_psnr[i + 2] = psnr(frame3_gt_img, img * 255)
    #     #
    #     # # backward prediction
    #     # img = warp(frame3_gt_img, backward_flow, backward_grad)
    #     # cv2.imwrite(
    #     #     "./Results/Multiscale_Lukas_kanade_results/corridor/backward_prediction/backward_pred_" + str(i) + ".pgm",
    #     #     img * 255)
    #     # backward_pred_scores.multiscale_lukas_kanade_ssim[i] = ssim(frame1_img, img * 255)
    #     # backward_pred_scores.multiscale_lukas_kanade_psnr[i] = psnr(frame1_img, img * 255)
    #
    #     # Interpolated
    #     # img = interpolate1(frame1_img, frame3_gt_img, forward_flow, backward_flow)
    #     # temp = "./Results/Multiscale_Lukas_kanade_results/corridor/interpolated_frame/interpolated_" + str(
    #     #     i + 1) + ".pgm"
    #     # cv2.imwrite(temp, img)
    #     # interpolated_frame_scores.multiscale_lukas_kanade_ssim[i + 1] = ssim(frame2_img, img)
    #     # interpolated_frame_scores.multiscale_lukas_kanade_psnr[i + 1] = psnr(frame2_img, img)
    #     print("Index", i, "Ending")
    # ## save scores in results/SSIM_PSNR_Scores/corridor

    #
    # os.makedirs("./"+result+"/SSIM_PSNR_Scores/corridor",exist_ok=True)
    # forward_pred_scores.to_csv("./"+result+"/SSIM_PSNR_Scores/corridor/forward_pred_scores.csv")
    # # backward_pred_scores.to_csv("./Results/SSIM_PSNR_Scores/corridor/backward_pred_scores4.csv")
    # # interpolated_frame_scores.to_csv("./Results/SSIM_PSNR_Scores/corridor/interpolated_frame_scores4.csv")
    #
    # print("Starting the Sphere Dataset...")
    # # Creating dataframe for scores (sphere dataset)
    # forward_pred_scores = pd.DataFrame(
    #     columns=["horn_schunk_ssim", "horn_schunk_psnr", "lukas_kanade_ssim", "lukas_kanade_psnr",], index=list(range(2, 19, 1)))
    #


    # Sphere Dataset
    for i in range(0, 17, 1):
        print("index", i, "starting")
        sph = "./Dataset/sphere/"
        frame1 = str("sphere.") + str(i) + str(".ppm")
        frame2 = str("sphere.") + str(i + 1) + str(".ppm")


        frame1_path = str(sph) + str(frame1)
        frame2_path = str(sph) + str(frame2)

        frame1_img = cv2.imread(frame1_path)
        frame2_img = cv2.imread(frame2_path)


        frame1_img = cv2.cvtColor(frame1_img, cv2.COLOR_BGR2GRAY)
        frame2_img = cv2.cvtColor(frame2_img, cv2.COLOR_BGR2GRAY)

        #create directory augmented_sphere_dataset_i if not present
        os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i),exist_ok=True)

        #create the directory as frame_i inside augmented_sphere_dataset_i
        os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i),exist_ok=True)

        #create the directory as frame_i+1 inside augmented_sphere_dataset_i
        os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i+1),exist_ok=True)

        #create the directory named as 0 inside augmented_sphere_dataset_i/frame_i
        os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i)+"/0",exist_ok=True)

        #create the directory named as 0 inside augmented_sphere_dataset_i/frame_i+1
        os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i+1)+"/0",exist_ok=True)

        #store the frame1_img in augmented_sphere_dataset_i/frame_i/0/ as 0.pgm
        cv2.imwrite("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i)+"/0/0.ppm",frame1_img)

        #store the frame2_img in augmented_sphere_dataset_i/frame_i+1/0/ as 0.pgm
        cv2.imwrite("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i+1)+"/0/0.ppm",frame2_img)

        OF = dummy_of(frame1_img)

        #loop for generating 10 augmented images
        for j in range(1,11,1):
            #copy the frame1_img and frame2_img to frame1_img_temp and frame2_img_temp
            frame1_img_temp = frame1_img.copy()
            frame2_img_temp = frame2_img.copy()

            frame1_img_temp,pos = put_rec(frame1_img_temp)
            black_pixel = 0

            #loop over the values of pos
            for m in range(pos[0][0],pos[1][0]):
                for n in range(pos[0][1],pos[1][1]):
                    # round up the OF float values to int
                    x = int(round(OF[m][n][0]))
                    y = int(round(OF[m][n][1]))
                    new_i = m + x
                    new_j = n + y
                    # if new_i and new_j are within the image
                    if new_i >= 0 and new_i < frame1_img_copy.shape[0] and new_j >= 0 and new_j < frame1_img_copy.shape[1]:
                        # typecase new_i and new_j to int
                        new_i = int(new_i)
                        new_j = int(new_j)
                        # replace the pixel at new_i,new_j of frame2_img_copy with black pixel
                        frame2_img_copy[new_i][new_j] = 0

            #create the directory names as j inside augmented_sphere_dataset_i/frame_i
            os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i)+"/"+str(j),exist_ok=True)

            #create the directory names as j inside augmented_sphere_dataset_i/frame_i+1
            os.makedirs("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i+1)+"/"+str(j),exist_ok=True)

            #store the frame1_img_temp in augmented_sphere_dataset_i/frame_i/j/ as j.ppm
            cv2.imwrite("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i)+"/"+str(j)+"/"+str(j)+".ppm",frame1_img_temp)

            #store the frame2_img_temp in augmented_sphere_dataset_i/frame_i+1/j/ as j.ppm
            cv2.imwrite("./"+result+"/augmented_sphere_dataset_"+str(i)+"/frame_"+str(i+1)+"/"+str(j)+"/"+str(j)+".ppm",frame2_img_temp)

    #print message as done
    print("Completed")
    #     # Calculating flow using hornshunk
    #     print("Calculating flow using hornshunk")
    #     forward_flow, forward_grad = horn_schunk_flow(frame1_img, frame2_img, lambada, max_iter, epsilon)
    #     # backward_flow, backward_grad = horn_schunk_flow(frame2_gt_img, frame1_img, lambada, max_iter, epsilon)
    #
    #     # forward prediction
    #     print("forward prediction")
    #     predicted_img = warp(frame2_img, forward_flow, forward_grad)
    #
    #     print("Writing Predicted Image")
    #     os.makedirs("./" + result + "/Horn_schunk_results/sphere/forward_prediction", exist_ok=True)
    #     cv2.imwrite("./" + result + "/Horn_schunk_results/sphere/forward_prediction/HSK_Sphere_forward_pred_" + str(i + 1) + ".ppm",
    #                 predicted_img * 255)
    #
    #     print("Calculating SSIM and PSNR")
    #     forward_pred_scores.horn_schunk_ssim[i + 2] = ssim(frame3_gt_img, predicted_img * 255)
    #     forward_pred_scores.horn_schunk_psnr[i + 2] = psnr(frame3_gt_img, predicted_img * 255)
    #
    #
    #     # Calculating flow using lukas kanade
    #     print("Calculating Optical Flow using Lucas Kanade method")
    #     forward_flow, forward_grad = lukas_kanade_flow(frame1_img, frame2_img, 9)
    #     # backward_flow, backward_grad = lukas_kanade_flow(frame3_gt_img, frame1_img, 9)
    #
    #     # forward prediction
    #     print("Predicting Image")
    #     predicted_img = warp(frame2_img, forward_flow, forward_grad)
    #
    #     print("Predicting Image")
    #     os.makedirs("./"+result+"/Lukas_kanade_results/sphere/forward_prediction", exist_ok=True)
    #     cv2.imwrite("./"+result+"/Lukas_kanade_results/sphere/forward_prediction/LK_Sphere_forward_pred_" + str(i + 2) + ".ppm",
    #                 predicted_img * 255)
    #
    #     print("Calculating SSIM and PSNR")
    #     forward_pred_scores.lukas_kanade_ssim[i + 2] = ssim(frame3_gt_img, predicted_img * 255)
    #     forward_pred_scores.lukas_kanade_psnr[i + 2] = psnr(frame3_gt_img, predicted_img * 255)
    #
    # ## Save scores for sphere dataset
    # os.makedirs("./" + result + "/SSIM_PSNR_Scores/sphere", exist_ok=True)
    # forward_pred_scores.to_csv("./"+result+"/SSIM_PSNR_Scores/sphere/forward_pred_scores.csv")
    # backward_pred_scores.to_csv("./Results/SSIM_PSNR_Scores/sphere/backward_pred_scores4.csv")
    # interpolated_frame_scores.to_csv("./Results/SSIM_PSNR_Scores/sphere/interpolated_frame_scores4.csv")
