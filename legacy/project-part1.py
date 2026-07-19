#project - part 1

import numpy as np
import cv2 

# create videocapture
cap = cv2.VideoCapture('output.mp4')

# create Background Subtractor objects
backSub = cv2.createBackgroundSubtractorKNN()

# Read Map Image
fmap = cv2.imread('2D_field.png')
fmap = cv2.resize(fmap,(1050,680))

# 9 correspondences for perspective transform
points1 = np.array([(144,166),
                    (1136,116),
                    (873,780),
                    (639,110),
                    (673,200),
                    (490,210),
                    (857,192),
                    (660,162),
                    (692,251)], dtype=np.int32)

points2 = np.array([(164,147),
                    (886,147),
                    (525,676),
                    (525,4),
                    (525,340),
                    (430,340),
                    (618,340),
                    (525,250),
                    (525,430)],   dtype=np.int32)   

# find perspective transform
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)

while True:
    
    # read video frame by frame
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    # apply GaussianBlur 
    m = 5
    frame2 = cv2.GaussianBlur(frame,(m,m),0)
    cv2.line(frame2,(0,120),(1280,57),color=(0,0,0),thickness=40)

    # update the background model (get foreground mask)
    fgMask = backSub.apply(frame2)
    
        
    # threshold for removing shadows
    ret, fgMask = cv2.threshold(fgMask,128,255,cv2.THRESH_BINARY)
    
    #opening
    kernel = np.array([ [0,0,0,1,1,0,0,0],
                        [0,0,0,1,1,0,0,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,1,1,1,1,0,0]], dtype='uint8')
        
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        
        
    #closing
    kernel = np.array([ [0,0,0,1,1,0,0,0],
                        [0,0,0,1,1,0,0,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,1,1,1,1,1,1,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,1,1,1,1,0,0],
                        [0,0,1,1,1,1,0,0]], dtype='uint8')
    
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
    #detection
    n,C,stats, centroids = cv2.connectedComponentsWithStats(fgMask);
    
    fr = frame.copy()    
    f = fmap.copy()
    
    
    for i in range(1,n):
    
        if stats[i,cv2.CC_STAT_TOP]<=240:
            if stats[i,cv2.CC_STAT_AREA]>20:
                point = np.array([centroids[i,0],centroids[i,1]+stats[i,3]/2]).reshape(-1,1,2).astype(np.float32)
                dst2 = cv2.perspectiveTransform(point,H).reshape(1,2)
                dst2 = dst2.astype('int32')
                cv2.circle(f,(dst2[0,0], dst2[0,1]), radius=7, color=[0,0,255], thickness=-1)
                cv2.circle(fr,(int(centroids[i,0]), int(centroids[i,1])), radius=10, color=[0,0,255], thickness=-1)
                
                
        elif 240<stats[i,cv2.CC_STAT_TOP]<=480:
            if stats[i,cv2.CC_STAT_AREA]>400:
                point = np.array([centroids[i,0],centroids[i,1]+stats[i,3]/2]).reshape(-1,1,2).astype(np.float32)
                dst2 = cv2.perspectiveTransform(point,H).reshape(1,2)
                dst2 = dst2.astype('int32')
                cv2.circle(f,(dst2[0,0], dst2[0,1]), radius=7, color=[0,0,255], thickness=-1)
                cv2.circle(fr,(int(centroids[i,0]), int(centroids[i,1])), radius=10, color=[0,0,255], thickness=-1)
       
        
        elif 480<stats[i,cv2.CC_STAT_TOP]:
            if stats[i,cv2.CC_STAT_AREA]>1600:
                point = np.array([centroids[i,0],centroids[i,1]+stats[i,3]/2]).reshape(-1,1,2).astype(np.float32)
                dst2 = cv2.perspectiveTransform(point,H).reshape(1,2)
                dst2 = dst2.astype('int32')
                cv2.circle(f,(dst2[0,0], dst2[0,1]), radius=7, color=[0,0,255], thickness=-1)
                cv2.circle(fr,(int(centroids[i,0]), int(centroids[i,1])), radius=10, color=[0,0,255], thickness=-1)
    
   
    # show the current frame ,the fg masks and Map
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', fr)
    cv2.namedWindow('FG Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('FG Mask', fgMask)
    cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
    cv2.imshow('Map', f)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    
