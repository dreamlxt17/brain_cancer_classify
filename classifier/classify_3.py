# coding=utf-8
# Author: Didia
# Date: 18-4-1
from sklearn.preprocessing import StandardScaler
import numpy as np
from models import RF, GB, KNN, SV_M, find_para
from glob import glob
from random import shuffle
from sklearn.feature_selection import SelectFromModel
from metric import comput_recall, comput_specificity, comput_residual
from sklearn.externals import joblib
from classify import read_data, classify
from classify import pred_person as ppp
from models import GB
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from sklearn.metrics import roc_curve, auc
from scipy import interp
from models import split_fold
from metric import comput_residual


def data_by_name(all_slices, name_list):
    '''按照name_list划分训练集'''
    slices= []
    for j, name in enumerate(name_list):
        slices.extend( [i for i in all_slices if name in i])
    return slices


def get_five_data(split, all_slices):
    '''按照split划分五折交叉验证训练集和测试集'''
    tr_names = [names[i] for i in split[0]]
    te_names = [names[i] for i in split[1]]
    tr_paths = []
    te_paths = []
    for name in tr_names:
        indices = [i for i, n in enumerate(all_slices) if name in n]  # 获取与该名字相关的所有slice
        tr_paths.extend([all_slices[i] for i in indices])
    for name in te_names:
        indices = [i for i, n in enumerate(all_slices) if name in n]  # 获取与该名字相关的所有slice
        te_paths.extend([all_slices[i] for i in indices])
    shuffle(tr_paths)
    shuffle(te_paths)
    return tr_paths, te_paths


def plot_ROC():
    '''联合诊断五折交叉验证ROC曲线'''
    # dframe = pandas.read_excel('/home/didia/Didia/Brain_data/co-worker.xlsx', sheet_name='sheet2')
    import xlrd
    dataframe = xlrd.open_workbook('/home/didia/Didia/Brain_data/co-worker.xlsx')
    sheet1 = dataframe.sheet_by_index(0)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i in [0,2,4,6,8]:
        truth = sheet1.col_values(i)[1:-2]
        predict = sheet1.col_values(i+1)[1:-2]
        print truth
        print predict

        fpr, tpr, thresholds = roc_curve(truth, predict)

        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= 5  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值

    # return mean_fpr, mean_tpr, mean_auc

    # 画对角线
    # plt.subplot(subplt)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Co-diagnosis - Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

import xlrd
dataframe = xlrd.open_workbook('/home/didia/Didia/Brain_data/machine_names.xlsx')
sheet1 = dataframe.sheet_by_index(0)

nrows = sheet1.nrows
edema_dict = {'1':[], '2':[], '3':[], '4':[]}
enhanced_dict = {'1':[], '2':[], '3':[], '4':[]}
a = []
for i in range(1, nrows):
    row_data = sheet1.row_values(i)
    edema_dict[str(int(row_data[3]))].append(row_data[2].encode('raw_unicode_escape').strip())
    enhanced_dict[str(int(row_data[4]))].append(row_data[2].encode('raw_unicode_escape').strip())
def four_machine_roc(edema_dict, all_slices, subtype='edema'):
    '''四台机器交叉验证'''
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    edema_split =  edema_dict.values()

    edema_names = [[edema_split[0], edema_split[1]+edema_split[2]+edema_split[3]]]
    edema_names.append( [edema_split[1], edema_split[0]+edema_split[2]+edema_split[3]])
    edema_names.append( [edema_split[2], edema_split[0]+edema_split[1]+edema_split[3]])
    edema_names.append([edema_split[3], edema_split[0]+edema_split[1]+edema_split[2]])

    m = 4
    residual = []
    performance = [0] * 8
    for i, names in enumerate(edema_names):
        tr_paths, te_paths = data_by_name(all_slices, names[1]), data_by_name(all_slices, names[0])
        tr_slice, tr_data, tr_label, tr_area = read_data(tr_paths, num_=116)
        te_slice, te_data, te_label, te_area = read_data(te_paths, num_=116)

        #     return trainacc, testacc, tr_acc, te_acc, tr_recall, te_recall, tr_speci, te_speci,prb_2, l2
        result = \
            classify(tr_data, te_data, tr_label, te_label, tr_slice, te_slice, tr_area, te_area, clf=GB())

        fpr, tpr, thresholds = roc_curve(result[-1], result[-2])
        residual.append(comput_residual(result[-1], result[-2]))

        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        performance[0] += result[0]
        performance[1] += result[1]
        performance[2] += result[2]
        performance[3] += result[3]
        performance[4] += result[4]
        performance[5] += result[5]
        performance[6] += result[6]
        performance[7] += result[7]

    print 'train/slice', performance[0] / m
    print 'test/slice', performance[1] / m
    print 'train/person', performance[2] / m
    print 'test/person', performance[3] / m
    print 'train/recall', performance[4] / m
    print 'test/recall', performance[5] / m
    print 'train/speci', performance[6] / m
    print 'test/speci', performance[7] / m
    print np.mean(residual, axis=0)

    mean_tpr /= 5  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值

    # return mean_fpr, mean_tpr, mean_auc

    # 画对角线
    # plt.subplot(subplt)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} - Receiver operating characteristic'.format(subtype))
    plt.legend(loc="lower right")
    plt.show()


def five_fold_roc(spliting, all_slices, subtype, random_flag=True, clf=GB()):
    '''随机五折分组交叉验证'''
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # all_tpr = []
    residual = []
    performance = [0]*8

    m = 5
    for i, split in enumerate(spliting):
        if random_flag: # 随机分配
            tr_paths, te_paths = get_five_data(split, all_slices)
        else:   # 按固定名字分组
            tr_paths, te_paths = data_by_name(all_slices, split[0]), data_by_name(all_slices, split[1])

        tr_slice, tr_data, tr_label, tr_area = read_data(tr_paths, num_=116)
        te_slice, te_data, te_label, te_area = read_data(te_paths, num_=116)

        #     return trainacc, testacc, tr_acc, te_acc, tr_recall, te_recall, tr_speci, te_speci,prb_2, l2
        result = \
            classify(tr_data, te_data, tr_label, te_label, tr_slice, te_slice, tr_area, te_area, clf=GB())

        fpr, tpr, thresholds = roc_curve(result[-1], result[-2])
        residual.append(comput_residual(result[-1], result[-2]))

        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        performance[0] += result[0]
        performance[1] += result[1]
        performance[2] += result[2]
        performance[3] += result[3]
        performance[4] += result[4]
        performance[5] += result[5]
        performance[6] += result[6]
        performance[7] += result[7]

    print 'train/slice', performance[0] / m
    print 'test/slice', performance[1] / m
    print 'train/person', performance[2] / m
    print 'test/person', performance[3] / m
    print 'train/recall', performance[4] / m
    print 'test/recall', performance[5] / m
    print 'train/speci', performance[6] / m
    print 'test/speci', performance[7] / m
    print np.mean(residual, axis=0)

    mean_tpr /= 5  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值

    # return mean_fpr, mean_tpr, mean_auc

        # 画对角线
    # plt.subplot(subplt)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} - Receiver operating characteristic'.format(subtype))
    plt.legend(loc="lower right")
    plt.show()




if __name__ =='__main__':
    # from name_config import names_list
    # from config import names_list
    type = 116
    if_norm = '_norm'
    print type

    feature_dir = '/home/didia/Didia/Brain_data/{}_feature_116{}'.format('edema', if_norm) # 67, 83, 127,116
    edema_slices = glob(feature_dir + '/*/*.npy')  #+ glob(post_dir + '/*/*.npy')

    feature_dir = '/home/didia/Didia/Brain_data/{}_feature_116{}'.format('enhanced', if_norm)  # 67, 83, 127,116
    enhanced_slices = glob(feature_dir + '/*/*.npy')  # + glob(post_dir + '/*/*.npy')

    names = set(s.split('/')[-1].split('-')[0] for s in enhanced_slices)
    names = [n for n in names]
    print len(names)

    # print '===================================================四台机器交叉验证水肿结果--------------------'
    # four_machine_roc(edema_dict, edema_slices, subtype='edema')
    # print '===================================================四台机器交叉验证增强结果--------------------'
    # four_machine_roc(enhanced_dict, enhanced_slices, subtype='enhanced')

    # plot_ROC()

    # # 水肿增强各自随机五折交叉验证
    # # '''
    # spliting = split_fold(len(names))
    # print '===================================================水肿结果--------------------'
    # five_fold_roc(spliting, edema_slices, subtype='edema')
    # print '===================================================增强结果--------------------'
    # five_fold_roc(spliting, enhanced_slices, subtype='enhanced')
    # # '''

    # 按划分好的名字交叉验证
    from config import split_names
    spliting = split_names
    print '===================================================水肿结果--------------------'
    five_fold_roc(spliting, edema_slices, subtype='edema', random_flag=False)
    print '===================================================增强结果--------------------'
    five_fold_roc(spliting, enhanced_slices, subtype='enhanced', random_flag=False)



    #
    # names_1 = set(s.split('/')[-1].split('-')[0] for s in enhanced_slices)
    # names_1 = [n for n in names_1]
    # for n in names:
    #     if n not in names_1:
    #         print n
    # print len(names_1), len(names)
    # acc = 0
    # for i in range(50):
    #     spliting = split_fold(len(names))
    #     acc+=five_fold_roc(spliting=spliting, all_slices= edema_slices)
    # print acc/50
    # five_fold_roc(spliting=spliting, all_slices=enhanced_slices)



    # save_model(all_slices)
    # pred_with_model(post_dir)

    # slice_list, data, label, area = read_data(
    #                                  glob('/all/DATA_PROCEING/preprocessed/edema_feature' +'/*/*'))  # 430 来自segmentation_train

    # param_test1 = {'max_depth': range(1,13,1), 'n_estimators': range(20,200,10)}
    # find_para(data, label, clf=GB(), parameters=param_test1)

