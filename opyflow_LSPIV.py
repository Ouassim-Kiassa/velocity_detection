import os
import cv2
import numpy as np

import sys
sys.path.append("opyflow/src")
import opyf




# Loading video and set parameters

video = opyf.videoAnalyzer("video.mp4")
video.set_vecTime(Ntot=25,starting_frame=200)

video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)

#cv2.imwrite("mask.png",video.vis)

# Loading mask for stabilization

mask=cv2.imread('mask.png')
A=mask>100
video.set_stabilization(mask=A[:,:,0],mute=False)


## Orthorectification

image_points = np.array([
	(250, 1470),   # left
	(2161, 1180),   # right
	(130, 620),  # left front
	(1700, 200),   # right front
                            ], dtype="double")

# coordinates
model_points = np.array([
                    (224, 207, 0) ,#left 
                    (246, 221, 0),# right bank
                    (232, 213, 0),    # left front bank
                    (252, 218, 0),], dtype="double")  # riĝht front bank


abs_or=model_points[0]
model_points = model_points - model_points[0]


video.set_birdEyeViewProcessing(image_points,
                                model_points, [1, 1, -30.],
                                rotation=np.array([[1., 0, 0],[0,-1,0],[0,0,-1.]]),
                                scale=True,framesPerSecond=30)


print('Processing started...')


video.set_vlim([0, 10])

video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='quiver',display2='field',displayColor=True)
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15,range_Vx=[0.01,10])

video.filterAndInterpolate()

print('Processing started...')