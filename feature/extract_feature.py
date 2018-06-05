# coding=utf8

import cv2
from libs.basic import *
# from libs.glcm import *
import matplotlib
import matplotlib.pyplot as plt

def main():
    image_path = '../data/images/1.jpg'
    im = surf(image_path)
    cv2.imshow('SURF_features',im)
    cv2.waitKey()




if __name__ == '__main__':
    main()
