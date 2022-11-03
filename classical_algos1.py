
import numpy as np
import scipy.ndimage
from Functions.grdient import *
import cv2 

#parameters for classical optical flow algorithms
HS_ITER = 400
HS_LAMBDA = 2
HS_EPSILON = 0.001
LS_N = 5


def horn_schunk_flow(img_pth_0,img_pth_2,lambada = HS_LAMBDA,max_iter = HS_ITER,epsilon= HS_EPSILON):     #img0,img2
    """

    :param img0: first frame
    :param img2: second frame
    :param lambada: hyper parameter
    :param max_iter: threshold for iterations
    :param epsilon: decay rate
    :return: flow and gradient
    """
    decay=10000
    i=0
    ## averaging kernel
    avg_kernel=np.array([[0,1,0],[1,0,1],[0,1,0]])/4


    #load img0 from the path
    img0=cv2.imread(img_pth_0)
    img0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

    #load img2 from the path
    img2=cv2.imread(img_pth_2)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #using scipy.ndimage.imread, then convert to numpy array
    # img0 = np.array(scipy.ndimage.imread(img_pth_0, flatten=True))
    ## Calculating gradient
    fx,fy,ft=grad_cal(img0,img2)

    a=np.zeros((img0.shape))
    b=np.zeros((img0.shape))

    while(decay>epsilon and i<=max_iter):
        i+=1
        ## Calculating
        a_avg = scipy.ndimage.convolve(input=a, weights=avg_kernel)
        b_avg = scipy.ndimage.convolve(input=b, weights=avg_kernel)

        temp = (fx * a_avg + fy * b_avg + ft) / (1+lambada*( fx ** 2 + fy ** 2))

        ## Updating flow
        a=a_avg-lambada*fx*temp
        b=b_avg-lambada*fy*temp

        ## calculating decay
        decay=np.max(np.max((abs(a-a_avg)+abs(b-b_avg))))
        #print(i,decay)

    #create a numpy ndarray to store the flow of two channels
    output_flow = np.zeros((img0.shape[0], img0.shape[1], 2))
    output_flow[:, :, 0] = a
    output_flow[:, :, 1] = b
        #stack the two channels to form a 2-channel flow
        # output_flow = np.stack((a, b), axis=2)

    return output_flow



def lukas_kanade_flow(img_pth_0,img_pth_2,N = LS_N):
    """
    :param img0: first image
    :param img2: second image (next frame)
    :param N: size of the window (no. of equations= N**2)
    :return: flow a,b and gradients fx, fy, ft
    """

     #load img0 from the path
    img0=cv2.imread(img_pth_0)
    img0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

    #load img2 from the path
    img2=cv2.imread(img_pth_2)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    #Initializing flow with zero matrix
    a=np.zeros((img0.shape))
    b=np.zeros((img0.shape))

    # Calculating gradients
    fx,fy,ft=grad_cal(img0,img2)

    for x in range(N//2,img0.shape[0]-N//2):
        for y in range(N//2,img0.shape[1]-N//2):

            ## Selecting block(Window) around the pixel
            block_fx = fx[x - N // 2:x + N //2 + 1,  y - N // 2:y + N // 2 + 1]
            block_fy = fy[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]
            block_ft = ft[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]

            ## Flattening to genrate equations
            block_ft = block_ft.flatten()
            block_fy = block_fy.flatten()
            block_fx = block_fx.flatten()

            ## Reshaping to generate the format of Ax=B
            B=-1*np.asarray(block_ft)
            A=np.asarray([block_fx,block_fy]).reshape(-1,2)

            ## Solving equations using pseudo inverse
            flow=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)),np.transpose(A)),B)

    #         ## Updating flow matrix a,b
    #         a[x,y]=flow[0]
    #         b[x,y]=flow[1]

    # return [a,b],[fx,fy,ft]
    #create a numpy ndarray to store the flow of two channels
    output_flow = np.zeros((img0.shape[0], img0.shape[1], 2))
    output_flow[:, :, 0] = a
    output_flow[:, :, 1] = b
        #stack the two channels to form a 2-channel flow
        # output_flow = np.stack((a, b), axis=2)

    return output_flow