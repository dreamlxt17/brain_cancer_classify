# -*- coding: utf-8 -*-

import cv2
import numpy as np  

def surf(image_path):
    im = cv2.imread(image_path)
    # cv2.imshow('original',im)
    # 下采样
    # im_lowers = cv2.pyrDown(im) 
    # cv2.imshow('im_lowers',im_lowers)

    # 检测特征点
    # s = cv2.SIFT() # 调用SIFT
    s = cv2.SURF() # 调用SURF
    keypoints = s.detect(im)

    # 显示特征点
    for k in keypoints:
        cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),1,(0,255,0),-1)
        # cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)
            
    return im

def orb():
    # 基于ORB特征提取算法图像匹配 python实现

    img1 = cv2.imread('test1.png')  
    img2 = cv2.imread('test12.png')  
  
    #最大特征点数,需要修改，5000太大。  
    orb = cv2.ORB_create(5000)  
  
    kp1, des1 = orb.detectAndCompute(img1,None)  
    kp2, des2 = orb.detectAndCompute(img2,None)  
  
    #提取并计算特征点  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)  
    #knn筛选结果  
    matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)  
    good = [m for (m,n) in matches if m.distance < 0.75*n.distance]  
    #查看最大匹配点数目  
    print len(good) 
