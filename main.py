# coding=utf8

'''
edit in 2017-12-18 by jin
'''

from libs import glcm
#from libs import hog
import glob
import numpy as np
from matplotlib import pyplot as plt
import mahotas
import cv2
from radiomics import featureextractor
import radiomics
import SimpleITK as sitk

def deskew(img):
    SZ = 20
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    print(img.shape)
    return img

# 保存hog到np文件 主要参数
def save_hog(roi_data_dir, save_data_dir, dir_0_1):
    #save_dir = save_data_dir + 'hog/' + data_dir + '/' + dir_0_1 + '/'
    save_dir = save_data_dir + 'hog/' + dir_0_1 + '/'
    all_roi_data = get_all_roi_data(roi_data_dir, dir_0_1)
    # file_name 0_BAI_WEN_TANG-14 roi_data (1/2/3, 256, 256)
    winSize = (512, 512)
    blockSize = (512, 512)
    blockStride = (256, 256)
    cellSize = (256, 256)
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
    for i, (file_name, roi_data) in enumerate(all_roi_data.items()):
        roi_li = []
        for roi in roi_data:
            # convert into np.uint8
            roi = roi.astype(np.uint8)
            #roi = deskew(roi)
            descriptor = hog.compute(roi)
            descriptor = np.squeeze(descriptor)
            roi_li.append(descriptor)
        roi_arr = np.asarray(roi_li)
        np.save(save_dir + file_name, roi_arr)
        # if i > 10:
        #     break
    pass

# use mahotas
def save_haralick(roi_data_dir, save_data_dir, dir_0_1):
    #save_dir = save_data_dir + 'haralick/' + data_dir + '/' + dir_0_1 + '/'
    save_dir = save_data_dir + 'haralick/' + dir_0_1 + '/'
    all_roi_data = get_all_roi_data(roi_data_dir, dir_0_1)
    # file_name 0_BAI_WEN_TANG-14 roi_data (1/2/3, 256, 256)
    for (file_name, roi_data) in all_roi_data.items():
        roi_li = []
        for roi in roi_data:
            roi = roi.astype(np.uint8)
            roi_li.append(mahotas.features.haralick(roi).mean(0))
        roi_arr = np.asarray(roi_li)
        np.save(save_dir + file_name, roi_arr)
        #break
    pass

# use pyradiomics glcm, rlm

def  get_original_data(original_data_dir, slice, dir_0_1, file_name):
    original_data_dir = original_data_dir + slice
    original_data = []
    edema_data = np.load(original_data_dir + '/edema/' + dir_0_1 + '/' + file_name + '.npy')
    original_data.append(edema_data)
    enhanced_data = np.load(original_data_dir + '/enhanced/' + dir_0_1 + '/' + file_name + '.npy')
    original_data.append(enhanced_data)
    necrosis_data = np.load(original_data_dir + '/necrosis/' + dir_0_1 + '/' + file_name + '.npy')
    original_data.append(necrosis_data)
    return original_data
'''
    original_glrlm_ShortRunLowGrayLevelEmphasis : 0.00822976624416
	original_glrlm_GrayLevelVariance : 39.118151022
	original_glrlm_LowGrayLevelRunEmphasis : 0.00860039789166
	original_glrlm_GrayLevelNonUniformityNormalized : 0.0451412381498
	original_glrlm_RunVariance : 0.0847945778959
	original_glrlm_GrayLevelNonUniformity : 175.635192315
	original_glrlm_LongRunEmphasis : 1.22684403826
	original_glrlm_ShortRunHighGrayLevelEmphasis : 268.974179841
	original_glrlm_RunLengthNonUniformity : 3500.04323157
	original_glrlm_ShortRunEmphasis : 0.955939173141
	original_glrlm_LongRunHighGrayLevelEmphasis : 341.286579098
	original_glrlm_RunPercentage : 0.940406463249
	original_glrlm_LongRunLowGrayLevelEmphasis : 0.0106011704787
	original_glrlm_RunEntropy : 4.91503800316
	original_glrlm_HighGrayLevelRunEmphasis : 281.066493909
	original_glrlm_RunLengthNonUniformityNormalized : 0.895049465948
'''

# 在某一方向具有相同灰度值的像素个数称为行程长度 灰度行程矩阵
def save_rlm(original_data_dir, roi_data_dir, save_data_dir, slice, data_dir, dir_0_1):
    #save_dir = save_data_dir + 'rlm/' + data_dir + '/' + dir_0_1 + '/'
    save_dir = save_data_dir + 'rlm/' + dir_0_1 + '/'
    all_roi_data = get_all_roi_data(roi_data_dir, dir_0_1)
    # file_name 0_BAI_WEN_TANG-14 roi_data (1/2/3, 256, 256)
    # 求出所有的特征，包括glrlm,glcm等
    params = './libs/rlm_params.yaml'
    extractor = featureextractor.RadiomicsFeaturesExtractor(params)
    for i, (file_name, roi_data) in enumerate(all_roi_data.items()):
        roi_li = []
        original_data = get_original_data(original_data_dir, slice, dir_0_1, file_name)
        # original_data (3, 256, 256) roi_data (3, 256, 256)
        for original, roi in zip(original_data, roi_data):
            original = original[np.newaxis, :]
            original = sitk.GetImageFromArray(original)
            # roi convert from 0-255 to 0-1
            roi[roi > 0] = 1
            # (256, 256) -> (1, 256, 256)
            roi = roi[np.newaxis, :]
            roi = sitk.GetImageFromArray(roi)
            #print(original, roi)
            result = extractor.execute(original, roi)
            # one slice glrlm feature
            glrlm_feature = []
            for key, value in result.items():
                if 'glrlm' in key:
                    glrlm_feature.append(value)
            # original data all 0, so label (1) not present in mask
            if  glrlm_feature == []:
                glrlm_feature = np.zeros(16)
            # 将numpy里的nan及inf更改为0
            if np.isnan(glrlm_feature).sum() > 0 or np.isinf(glrlm_feature).sum() > 0:
                # where_are_nans = np.isnan(glrlm_feature)
                # where_are_infs = np.isinf(glrlm_feature)
                # glrlm_feature[where_are_nans] = 0
                # glrlm_feature[where_are_infs] = 0
                glrlm_feature = np.zeros(16)
            roi_li.append(glrlm_feature)
        roi_arr = np.asarray(roi_li)
        np.save(save_dir + file_name, roi_arr)
        # if i > 2:
        #     break
    pass

def save_glcm(roi_data_dir, save_data_dir, dir_0_1):
    #save_dir = save_data_dir + 'glcm/' + data_dir + '/' + dir_0_1 + '/'
    save_dir = save_data_dir + 'glcm/' + dir_0_1 + '/'
    all_roi_data = get_all_roi_data(roi_data_dir, dir_0_1)
    # file_name 0_BAI_WEN_TANG-14 roi_data (1/2/3, 256, 256)
    for (file_name, roi_data) in all_roi_data.items():
        roi_li = []
        for roi in roi_data:
            roi_li.append(glcm.GLCM(roi).get_features())
        roi_arr = np.asarray(roi_li)
        np.save(save_dir + file_name, roi_arr)

# 去掉全黑的数据(如切片有水肿，无增强或坏死)

def get_all_roi_data(roi_data_dir, dir_0_1):
    #data_dir = roi_data_dir + slice +'/edema/'
    data_dir = roi_data_dir + 'edema/'
    #data_dir = roi_data_dir
    edema_file_names = glob.glob(data_dir + dir_0_1 + '/*.npy')
    enhanced_file_names = glob.glob(data_dir.replace('edema', 'enhanced') + dir_0_1 + '/*.npy')
    necrosis_file_names = glob.glob(data_dir.replace('edema', 'necrosis') + dir_0_1 + '/*.npy')
    edema_set = set(edema_file_names)
    enhanced_set = set([enhanced.replace('enhanced', 'edema') for enhanced in enhanced_file_names])
    # 注意数据 necrosis 格式
    necrosis_files = [necrosis.replace('necrosis_', 'edema') for necrosis in necrosis_file_names]
    necrosis_set = set([necrosis.replace('necrosis', 'edema') for necrosis in necrosis_files])
    # 取水肿，坏死，增强均有的切片
    #edema_file_names = list(edema_set & enhanced_set & necrosis_set)
    #edema_file_names = list(edema_set & enhanced_set)
    # 0 290 1 518 all 808
    print('number of data:', len(edema_file_names))
    all_roi_data = {}
    edema_file_names.sort()
    for file in edema_file_names:
        # flag 去掉含有全黑数据的切片 筛选数据
        flag = 1
        roi_data = []
        edema_data = np.load(file)
        if edema_data.max() <= 0.0:
            flag = 0
        roi_data.append(edema_data)
        if file.replace('edema', 'enhanced') in enhanced_file_names:
            enhanced_data = np.load(file.replace('edema', 'enhanced'))
            if enhanced_data.max() <= 0.0:
                flag = 0
            roi_data.append(enhanced_data)
        # 注意数据格式
        file = file.replace('-edema', '-edema_')
        if file.replace('edema', 'necrosis') in necrosis_file_names:
            necrosis_data = np.load(file.replace('edema', 'necrosis'))
            if necrosis_data.max() <= 0.0:
                flag = 0
            roi_data.append(necrosis_data)
        file_name = file.split('/')[-1].split('.')[0]
        roi_data = np.asarray(roi_data)
        # 不筛选是否全黑的数据
        flag = 1
        if flag:
            all_roi_data[file_name] = roi_data
        #all_roi_data[file_name] = roi_data
    # 0 114 长度。。1 278
    print('number of data(remove data which is all 0):', len(all_roi_data))
    return all_roi_data

def test():
    file = '/all/DATA_PROCEING/features/pre_208_feature/test/glcm/512/0/0-edema_TANG_XIAO_YUN_0.npy'
    data = np.load(file)
    print(data.shape, data)

    # data_dir = '/home/jin/pyradiomics/data/'
    # image_name = data_dir + 'brain1_image.nrrd'
    # mask_name = data_dir + 'brain1_label.nrrd'
    # image = sitk.ReadImage(image_name)
    # mask = sitk.ReadImage(mask_name)
    # image = np.asarray(image).reshape(-1, 256, 256)
    # mask = np.asarray(mask).reshape(-1, 256, 256)
    # for i,(ima, ma) in enumerate(zip(image, mask)):
    #     ax1 = plt.subplot(121)
    #     ax1.set_title('image')
    #     plt.imshow(ima)
    #     ax1 = plt.subplot(122)
    #     ax1.set_title('mask')
    #     plt.imshow(ma)
    #     plt.show()
    #     if i == 10:
    #         break
    pass

def save_shape_feature(roi_data_dir, save_data_dir, dir_0_1):
    save_dir = save_data_dir + 'shape/' + dir_0_1 + '/'
    all_roi_data = get_all_roi_data(roi_data_dir, dir_0_1)
    for i, (file_name, roi_data) in enumerate(all_roi_data.items()):
        roi_li = []
        for roi in roi_data:
            roi = roi.astype(np.uint8)
            # 纵横比  Aspect Ratio = (width / height)
            x, y, w, h = cv2.boundingRect(roi)
            rect_area = w * h
            # 部分数据全黑, 故分母可能为0, 当分母为0时，令aspect_ratio, extent 为0
            if h > 0:
                aspect_ratio = float(w) / h
                # 像素点的个数即为roi轮廓的面积
                #area = roi[roi > 0].size
                # 面积比率
                image_roi = roi
                img, contours, hierarchy = cv2.findContours(image_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # 轮廓 （82, 1, 2）
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                extent = float(area) / rect_area
                # 等效直径 Equivalent Diameter is the diameter of the circle whose area is same as the contour area
                equi_diameter = np.sqrt(4 * area / np.pi)
                # solidity is the ratio of contour area to its convex hull area.
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                # 坚固性
                if hull_area == 0:
                    solidity = 0
                else:
                    solidity = float(area) / hull_area
                mask = np.zeros(roi.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                # Mean Color or Mean Intensity 平均亮度(对比度) 最开始求出来是个含有4个元素的list.
                mean_val = np.mean(np.asarray(cv2.mean(roi, mask)))
                # 图像几何矩
                M = cv2.moments(cnt)
                # 从Hu不变矩定义图像的离心率 离心率表征了图像的最大轴向与图像的最小轴向的比率，满足物体平移，旋转，尺度不变性
                '''
                0阶矩（m00）:目标区域的质量 
                1阶矩（m01,m10）：目标区域的质心 
                2阶矩（m02,m11,m20）：目标区域的旋转半径 
                3阶矩（m03,m12,m21,m30）：目标区域的方位和斜度，反应目标的扭曲
                '''
                m20 = M['m20']
                m02 = M['m02']
                m11 = M['m11']
                if (m20 + m02) == 0:
                    eccentricity = 0
                else:
                    eccentricity = (m20 - m02 * m02 + 4 * m11 * m11) * 1.0 / ((m20 + m02) * (m20 + m02))
                # 粗糙度  perimeter contour / convex hull perimeter (转化计算方式)
                perimeter = cv2.arcLength(cnt, True)
                hullperimeter = cv2.arcLength(hull, True)
                if hullperimeter == 0:
                    roughness = 0
                else:
                    roughness = perimeter / hullperimeter
                # orientation is the angle at which object is directed.
                #(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            else:
                aspect_ratio = 0
                extent = 0
                equi_diameter = 0
                solidity = 0
                mean_val = 0
                eccentricity = 0
                roughness = 0
                angle = 0

            roi_li.append([aspect_ratio, extent, equi_diameter, solidity, mean_val, eccentricity, roughness])

        roi_arr = np.asarray(roi_li)
        np.save(save_dir + file_name, roi_arr)
        #break
        # if i > 10:
        #     break


def main():

    #roi_data_dir = '/all/DATA_PROCEING/preprocessed/data/'
    #roi_data_dir = '/all/DATA_PROCEING/preprocessed/predicted_roi_jin/train_e/'
    roi_data_dir = '/all/DATA_PROCEING/preprocessed/predicted_roi_jin/predict_7/predict_test_7/slice_512/'
    #original_data_dir = '/all/DATA_PROCEING/preprocessed/original/'
    #save_data_dir = '/all/DATA_PROCEING/features/jin_feature_data/'
    save_data_dir = '/all/DATA_PROCEING/features/predict_7/'
    #save_data_dir = '/all/DATA_PROCEING/features/train_test_feature_e/train/'
    #save_data_dir = '/all/DATA_PROCEING/features/all_feature_data/'

    # get hog feature vector (3, 36)
    save_hog(roi_data_dir, save_data_dir, '0')
    save_hog(roi_data_dir, save_data_dir, '1')

    #get_all_roi_data('0')

    # get haralick feature vector (3, 13)
    save_haralick(roi_data_dir, save_data_dir, '0')
    save_haralick(roi_data_dir, save_data_dir, '1')

    # get glrlm feature vector  (3, 16)
    # save_rlm(original_data_dir, roi_data_dir, save_data_dir, 'slice_512', '512', '0')
    # save_rlm(original_data_dir, roi_data_dir, save_data_dir, 'slice_512', '512', '1')


    # get shape feature
    # save_shape_feature(roi_data_dir, save_data_dir, '0')
    # save_shape_feature(roi_data_dir, save_data_dir, '1')


    # get glcm feature vector  (3, 11)  可以调distances参数
    # save_glcm(roi_data_dir, save_data_dir, '0')
    # save_glcm(roi_data_dir, save_data_dir, '1')
    #test()
    print("Done")
    pass


if __name__ == '__main__':
    main()