import os
import cv2
import numpy as np

import sys
sys.path.append("opyflow/src")
import opyf




# Loading video and set parameters

video = opyf.videoAnalyzer("video.mp4")

#cv2.imwrite("mask.png",video.vis)

# Loading mask for stabilization

mask=cv2.imread('mask.png')
A=mask>100
video.set_stabilization(mask=A[:,:,0],mute=True)


## Orthorectification

image_points = np.array([
	(250, 1470),   # left
	(2161, 1180),   # right
	(130, 630),  # left front
	(1700, 200),   # right front
                            ], dtype="double")

# coordinates
model_points = np.array([
                    (0, 180, 0) ,#left 
                    (0, 0, 0),# right bank
                    (91, 188, 0),    # left front bank
                    (93, 21, 0),], dtype="double")  # riĝht front bank


abs_or=model_points[1]
model_points = model_points - model_points[1]


video.set_birdEyeViewProcessing(image_points,
                                model_points, [26, 52, 164.],
                                rotation=np.array([[0., 0.5, 0],
                                                   [1, 0, 0],
                                                   [0, 0, 1]]),
                                scale=True, framesPerSecond=30, saveOutPut="birdeye")




video.set_vecTime(Ntot=10, shift=1, starting_frame=0)
video.set_vlim([0, 1])


video.set_filtersParams(maxDevInRadius=1, RadiusF=0.03, CLAHE = True)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate(display1='points',display2='field',displayColor=True)




video.opyfDisp.plotField(video.Field, vis=video.vis)
video.showXV(video.Xaccu, video.Vaccu)

print('Processing started...')
print('Processing started...2')