# coding=utf-8
''' get roi from original adn annotated data'''
import nipy
import dicom
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from glob import glob
import os
import numpy as np
from scipy import interpolate
from PIL import Image

shape = 256
origin_root ='/home/didia/Didia/Brain_data/total_original_data/'
roi_root = '/home/didia/Didia/Brain_data/total_ROI/'
save_dir = '/home/didia/Didia/Brain_data/clean_roi/'


list0 = glob(origin_root+'0/ede*')
list1 = glob(origin_root+'1/ede*')

names0 = [l.split('/')[-1].replace('edema', '') for l in list0]
names0.sort()
names1 = [l.split('/')[-1].replace('edema', '') for l in list1]
names1.sort()
print len(names0), len(names1)
total_names = names0+names1

roi_list = glob(roi_root+'/*.nii')
roi_list.sort()

def resize(con, ann, shape):
    con = Image.fromarray(con).resize([shape, shape])
    con = np.asarray(con)
    ann = Image.fromarray(ann).resize([shape, shape])
    ann = np.asarray(ann)
    return con, ann

def read_dcm(dcm_path):
    slices = {}
    for dcm in glob(dcm_path+'/*.dcm'):
        patdict = dicom.read_file(dcm)
        slices[patdict.InstanceNumber] = patdict.pixel_array
    slices = sorted(slices.items(), key=lambda d:d[0])
    slices = [s[1] for s in slices]
    return slices



def read_slice(name, dcm1, nii1, shape=512):
    name = ('_').join(name.split('_'))
    print name
    cls = dcm1.split('/')[-2]
    dcm2 = dcm1.replace('edema', 'enhanced')    #
    nii2 = nii1.replace('edema', 'enhanced')
    dir1 = save_dir + 'edema/' + cls + '/'
    dir2 = save_dir + 'enhanced/' + cls + '/'

    for d in [dir1, dir2]:
        if not os.path.exists(d):
            os.makedirs(d)

    inter = False
    mask1 = np.transpose(nipy.load_image(nii1), [2,1,0])
    mask2 = np.transpose(nipy.load_image(nii2), [2,1,0])
    if mask1.shape[1] != shape or mask2.shape[1] != shape : #or mask3.shape[1] != shape:
        inter=True

    dcms1 = read_dcm(dcm1)
    for i, (ann, dcm) in enumerate(zip(mask1, dcms1)):
        if np.max(ann):
            # print i+1
            if inter:
                dcm, ann = resize(dcm, ann, shape)
            # plt.subplot(121)
            # plt.imshow(dcm, cmap='gray')
            # plt.subplot(122)
            # plt.imshow(ann, cmap='gray')
            # plt.show()

            roi = dcm*ann
            np.save(dir1 +name+'-{}.npy'.format(i+1), roi)
            # print dir1 +name+'-{}.npy'.format(i+1)


    dcms2 = read_dcm(dcm2)
    for i, (ann, dcm) in enumerate(zip(mask2, dcms2)):
        if np.max(ann):
            # print i+1
            if inter:
                dcm, ann = resize(dcm, ann, shape)
            # plt.subplot(121)
            # plt.imshow(dcm, cmap='gray')
            # plt.subplot(122)
            # plt.imshow(ann, cmap='gray')
            # plt.show()

            roi = dcm*ann
            np.save(dir2 +name+'-{}.npy'.format(i+1), roi)

name_list = total_names
# name_list = ['HU_XIAO_RUI']
# name_list = ['QI_YUAN_YANG', 'SHEN_JIAN_SHE']  # 这两个病人有100多张enhance
# name_list = ['SHEN_JIAN_SHE']

def main(shape, type=0):
    # for name in name_list[type]:
    for name in name_list:
        # print name
        ede_dcm = origin_root + str(type) + '/edema' + name
        print ede_dcm
        ede_nii = ede_dcm.replace('total_original_data/{}'.format(type), 'total_ROI/') + '_Merge.nii'
        # ede_nii = ede_dcm.replace('post_test/{}'.format(type), 'post_ROI/') + '_Merge.nii'
        print ede_nii

        read_slice(name, ede_dcm, ede_nii, shape=shape)

if __name__=='__main__':

    main(shape)
    # pass





