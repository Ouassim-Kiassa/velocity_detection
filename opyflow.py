import sys
sys.path.append("opyflow/src")

import opyf
import cv2
import numpy as np

video = opyf.videoAnalyzer("video.mp4")




## Time Vector Setting

# The parameters [Ntot], [shift], [starting_frame], and [step] control the processing plan. 
# [Ntot] specifies the total number of image pairs
# [shift] specifies the shift between two pairs
# [starting_frame] specifies the first image
# [step] specifies the number of images between two images of each pair

print("Next, let's set the time vector for processing.")

video.set_vecTime(Ntot=10, shift=1, step=2, starting_frame=20)
print(video.vec, '\n', video.prev)


video.extractGoodFeaturesAndDisplacements(
    display='quiver', displayColor=True, width=0.002)


# Norm Velocity Limits
video.set_vlim([0, 30])

video.extractGoodFeaturesAndDisplacements(
    display='quiver', displayColor=True, width=0.002)

## Orthorectification

image_points = np.array([
	(250, 1470),   # left
	(2161, 1180),   # right
	(130, 620),  # left front
	(1700, 200)   # right front
                            ], dtype="double")

# coordinates
model_points = np . array ([
    (224 , 207 , 69) ,# left
    (246 , 221 , 52) ,# right bank
    (232 , 213 , 33) , # left front bank
    (252 , 218 , 18) ,] , dtype =" double ")


abs_or = model_points[0]
model_points = model_points - model_points[0]

video.set_birdEyeViewProcessing( image_points ,
                                model_points, [0.8 , 1.4 , -1.7],
                                rotation = np.array([[1. , 0 ,
                                0] ,[0 ,1 ,0] ,[0 ,0 ,1.]]) ,
                                scale = True , framesPerSecond =30)

