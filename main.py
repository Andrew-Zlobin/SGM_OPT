import cv2
import numpy as np
import time 
import matplotlib.pyplot as plt
from depth import depth_map
from configs import img_path1,img_path2
from disparity import disparitymap
from image import Image_processing,downsample_image,create_output

from sgm import create_disparity_map


def main():
    img=cv2.imread(img_path1,1)
    # img= downsample_image(img,1)
    imgL=Image_processing(img_path1)
    imgR=Image_processing(img_path2)
    plt.imshow(imgL, cmap='gray')
    plt.show()
    print("imgL shape = ", imgL.shape)
    
    # Map= disparitymap(imgL,imgR)
    # print("Map shape = ", Map.shape)
    Map = create_disparity_map(imgL,imgR)
    print(np.unique(Map))
    print("Map shape = ", Map.shape)
    # print(Map)

    img= downsample_image(img,1)#, Map.shape)

    coordinates= depth_map(Map,img)
    print('\n Creating the output file... \n')
    create_output(coordinates,'praxis.ply')
 
    

main()