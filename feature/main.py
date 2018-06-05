# coding=utf8

'''
获取所有roi对应的特征，并以切片格式保存
'''
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('TkAgg')
import mahotas
import radiomics
import SimpleITK as sitk
from libs import glcm
import os
import shutil
from scipy.ndimage import zoom
from radiomics import featureextractor
import dicom
import warnings
warnings.filterwarnings("ignore")

def normalize(roi_data):
    # 归一化为0-255
    i_max = np.max(roi_data)
    i_min = np.min(roi_data)
    return (roi_data-i_min)*255.0/(i_max-i_min)

def standardrize(roi_data):
    # 标准化，减去均值除以方差
    mean = np.mean(roi_data)
    var = np.var(roi_data)
    return (roi_data-mean)*1.0/var

def comput_rlm(roi, origin, extractor, shape=208):
    shape = origin.shape[-1]
    origin = center_crop(origin)

    original = origin[np.newaxis, :]
    original = sitk.GetImageFromArray(original)
    # roi convert from 0-255 to 0-1
    roi[roi > 0] = 1
    # (256, 256) -> (1, 256, 256)
    roi = roi[np.newaxis, :]
    roi = sitk.GetImageFromArray(roi)
    # print(original, roi)
    result = extractor.execute(original, roi)
    # glrlm = 16, glcm =
    # one slice glrlm feature
    glrlm = []
    glcm = []
    gldm = []
    glszm = []
    order = []
    shape = []

    for key, value in result.items():
        if 'glrlm' in key:   # 16
            glrlm.append(value)
        # if 'firstorder' in key:    # 23
        #     order.append(value)
        # if 'shape' in key:
        #     shape.append(value) # 13
    # original data all 0, so label (1) not present in mask
    if glrlm == []:
        glrlm = np.zeros(16)
    # if order == []:
    #     order = np.zeros(18)
    # if shape == []:
    #     shape = np.zeros(13)

    # 将numpy里的nan及inf更改为0
    if np.isnan(glrlm).sum() > 0 or np.isinf(glrlm).sum() > 0:
        glrlm = np.zeros(16)

    feature = np.hstack([glrlm,  order, shape])
    # return np.array([glrlm])
    feature = glrlm
    return np.array([feature])



def comput_hog(roi, shape=208):
    '''
    :param roi: 一个人的所有roi slice
    :return: 每张slice 36个 histogram 特征
    '''
    winSize = (shape, shape)
    blockSize = (shape, shape)
    blockStride = (shape/2, shape/2)
    cellSize = (shape/2, shape/2)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    slice = roi.astype(np.uint8)
    descriptor = hog.compute(slice)
    descriptor = np.squeeze(descriptor)
    return np.array([descriptor])

def comput_haralick(roi, shape=208):
    '''
    :param roi: 一个人的所有roi切片
    :return: 每张slice 13个harlick特征
    '''
    slice = roi.astype(np.uint8)
    descriptor = mahotas.features.haralick(slice).mean(0)
    return np.array([descriptor])

def comput_glcm(roi, shape=208):
    """
    :param roi: 一个人的所有roi切片
    :param shape:
    :return: 每张切片的 44 个特征
    """
    return np.array([glcm.GLCM(np.array(roi)).get_features()])

def comput_shape(roi, shape=208):
    '''
    计算形状特征
    :param roi: 计算形状特征
    :return: 每张slice 7个特征
    '''
    slice = roi.astype(np.uint8)
    # 纵横比  Aspect Ratio = (width / height)
    x, y, w, h = cv2.boundingRect(slice)
    rect_area = w * h
    # 部分数据全黑, 故分母可能为0, 当分母为0时，令aspect_ratio, extent 为0
    if h > 0:
        aspect_ratio = float(w) / h
        image_roi = slice
        img, contours, hierarchy = cv2.findContours(image_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        extent = float(area) / rect_area
        equi_diameter = np.sqrt(4 * area / np.pi)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            solidity = 0
        else:
            solidity = float(area) / hull_area
        mask = np.zeros(slice.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_val = np.mean(np.asarray(cv2.mean(slice, mask)))
        M = cv2.moments(cnt)
        m20 = M['m20']
        m02 = M['m02']
        m11 = M['m11']
        if (m20 + m02) == 0:
            eccentricity = 0
        else:
            eccentricity = (m20 - m02 * m02 + 4 * m11 * m11) * 1.0 / ((m20 + m02) * (m20 + m02))
        perimeter = cv2.arcLength(cnt, True)
        hullperimeter = cv2.arcLength(hull, True)
        if hullperimeter == 0:
            roughness = 0
        else:
            roughness = perimeter / hullperimeter
        # orientation is the angle at which object is directed.
        # (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    else:
        aspect_ratio = 0
        extent = 0
        equi_diameter = 0
        solidity = 0
        mean_val = 0
        eccentricity = 0
        roughness = 0

    return np.array([[aspect_ratio, extent, equi_diameter, solidity, mean_val, eccentricity, roughness]])

def center_crop(data, crop_size=[208, 208]):
    w, h = data.shape
    pw = (w - crop_size[0]) / 2
    ph = (h - crop_size[1]) / 2
    data = data[pw:crop_size[0] + pw, ph:crop_size[1] + ph]
    return data

def get_all_feature(data_dir, save_dir):
    edema_file_names = glob.glob(data_dir + '/*/*.npy') # 获取ROI数据
    edema_file_names.sort()

    print('number of data:', len(edema_file_names))
    all_roi_data = {}
    edema_file_names.sort()
    params = './libs/rlm_params.yaml'
    extractor = featureextractor.RadiomicsFeaturesExtractor(params)
    for i, file in enumerate(edema_file_names):
        print file, '--------------------------', i
        # file = '/all/DATA_PROCEING/preprocessed/new_roi/edema/1/WU_XU_BIN-15.npy'
        brain_type = file.split('/')[-3]
        label = file.split('/')[-2]
        name = file.split('/')[-1]
        person, idx = name.split('-')
        idx = int(idx.replace('.npy',''))# dcm从1开始计数
        roi_data = center_crop(np.load(file))
        print np.max(roi_data), np.min(roi_data)
        dcm_data = (dicom.read_file(origin_data_dir+label+'/{}'.format(brain_type)+person+'/'+str(idx)+'.dcm')).pixel_array # 水肿

        # roi_data = normalize(roi_data)
        # dcm_data = normalize(dcm_data)

        # roi_data = standardrize(roi_data)
        # dcm_data = standardrize(dcm_data)
        # print np.max(normalized_dcm), np.min(normalized_dcm)
        # plt.subplot(121)
        # plt.imshow(dcm_data, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(roi_data, cmap='gray')
        # plt.show()

        rlm = comput_rlm(roi_data, dcm_data, extractor) # 16
        glcm = comput_glcm(roi_data) # 44
        hog = comput_hog(roi_data)  # 36
        haralick = comput_haralick(roi_data) # 13
        shape = comput_shape(roi_data) # 7
        # # feature = np.hstack([glcm, hog, haralick, shape]) # 顺序一定不能乱
        feature = np.hstack([hog, haralick, shape, glcm, rlm]) # 顺序一定不能乱
        # feature = rlm
        # if feature == None:
        #     continue
        print feature.shape

        roi_data[roi_data!=0]=1
        area = np.sum(roi_data)
        print area
        np.save(save_dir + '/' + label +'/' +name+'.npy', [feature, area])
        # break
    return all_roi_data


if __name__ == '__main__':

    type = 'enhanced'
    size = 116
    # origin_data_dir = '/all/DATA_PROCEING/total_original_data/'
    # roi_data_dir = '/all/DATA_PROCEING/preprocessed/new_roi/{}/'.format(type) # or enhanced
    # save_dir = '/all/DATA_PROCEING/preprocessed/{}_feature_{}/'.format(type, str(size))

    origin_data_dir = '/home/didia/Didia/Brain_data/total_original_data/'
    # roi_data_dir = '/home/didia/Didia/Brain_data/new_roi/{}/'.format(type)  # or enhanced
    roi_data_dir = '/home/didia/Didia/Brain_data/clean_roi/{}/'.format(type)  # or enhanced
    # save_dir = '/home/didia/Didia/Brain_data/{}_feature_{}_norm/'.format(type, str(size))
    save_dir = '/home/didia/Didia/Brain_data/{}_feature_{}_origin/'.format(type, str(size))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/1/')
        os.makedirs(save_dir + '/0/')

    # origin_data_dir = '/all/DATA_PROCEING/post_test/'
    # roi_data_dir = '/all/DATA_PROCEING/preprocessed/post_roi/{}/'.format(type)  # or enhanced
    # save_dir = '/all/DATA_PROCEING/preprocessed/{}_post_{}/'.format(type, str(size))

    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    #     #删除非空文件夹
    # os.makedirs(save_dir + '/1/')
    # os.makedirs(save_dir + '/0/')
    get_all_feature(roi_data_dir, save_dir)


    print("Done")