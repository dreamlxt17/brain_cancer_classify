# coding=utf-8
from sklearn.preprocessing import StandardScaler
import numpy as np
from models import RF, GB, KNN, SV_M
from glob import glob
from random import shuffle
from models import find_para
from sklearn.feature_selection import SelectFromModel
from metric import comput_recall, comput_specificity, comput_residual
from sklearn.externals import joblib

from models import split_fold



def read_data(all_slices, num_=67):
    slice_list = []
    feature_list = []
    label_list = []
    area_list = []
    for slice in all_slices:

        feature, area = np.load(slice)
        if feature.shape != (1, num_):
            continue
        name = slice.split('/')[-1]
        slice_list.append(name)
        label = int(slice.split('/')[-2])

        label_list.append(label)
        feature_list.append(feature)
        area_list.append(area)

    feature_list = np.reshape(feature_list, [len(feature_list), num_])
    label_list = np.reshape(label_list, [len(feature_list)])
    return slice_list, feature_list, label_list, np.array(area_list)

def pred_person(slice_list, pred_list, area_list, label_list):
    names = list(set(s.split('-')[0] for s in slice_list))
    print len(names)
    names.sort()
    print names
    acc = 0
    pos = 0
    target_list = []
    prediction_list = []
    pred_prob_list = []
    for i, name in enumerate(names):
        indices = [i for i, n in enumerate(slice_list) if n.startswith(name)]  # 获取与该名字相关的所有slice
        preds = pred_list[indices]
        areas = area_list[indices]
        areas = areas*1.0/np.sum(areas)
        label = label_list[indices][0]
        pos+=label
        prediction = np.sum(preds*areas)
        pred_prob_list.append(prediction)

        if prediction>0.5:
            p = 1
        else:
            p = 0
        if p == label:
            acc+=1
        target_list.append(label)
        prediction_list.append(p)
    acc = acc*1.0/len(names)
    recall = comput_recall(target_list, prediction_list)
    speci = comput_specificity(target_list, prediction_list)
    # print recall, speci
    print pos, len(names)-pos
    print acc
    return acc, recall, speci, target_list, prediction_list, pred_prob_list

def classify(data1, data2, label1, label2, slice_list1, slice_list2, area_1, area_2, clf=GB()):
    scaler = StandardScaler()
    scaler.fit(np.vstack([data1, data2]))
    # scaler.fit(data1)
    data1 = scaler.transform(data1)
    data2 = scaler.transform(data2)

    clf.fit(data1, label1)
    # selection = SelectFromModel(clf, prefit=True)
    # data1 = selection.transform(data1)
    # data2 = selection.transform(data2)
    # print data1.shape, data2.shape
    # clf.fit(data1, label1)


    pred1 = clf.predict(data1)
    pred2 = clf.predict(data2)
    trainacc = np.sum(pred1 == label1) * 1.0 / len(data1)
    testacc = np.sum(pred2 == label2) * 1.0 / len(data2)

    print trainacc, testacc
    tr_acc, tr_recall, tr_speci, l1, p1, prb_1 = pred_person(slice_list1, pred1, area_1, label1)
    te_acc, te_recall, te_speci, l2, p2, prb_2 = pred_person(slice_list2, pred2, area_2, label2)

    print '真实值-------：', l2
    print '预测值-------：', p2
    print '预测概率-----：',prb_2

    # return trainacc, testacc, tr_acc, te_acc
    return trainacc, testacc, tr_acc, te_acc, tr_recall, te_recall, tr_speci, te_speci,prb_2, l2


def gt_names(edema_dir):
    edema_slices = glob(edema_dir + '/*/*.npy')
    shuffle(edema_slices)
    names = set(s.split('/')[-1].split('-')[0] for s in edema_slices)
    names = [n for n in names]
    # shuffle(names)
    m = 65
    tr_names = names[:m]
    te_names = names[m:]

    return tr_names, te_names

def save_model(all_slices, clf = GB()):
    _, data, label, _ = read_data(all_slices, num_=type)
    scaler = StandardScaler()
    scaler.fit(data)
    joblib.dump(scaler, '/all/DATA_PROCEING/classify_model/scaler_train.m')
    data = scaler.transform(data)
    clf.fit(data, label)
    joblib.dump(clf, '/all/DATA_PROCEING/classify_model/classify_train.m')
    print 'done!'

def pred_with_model(all_slices, clf=GB()):
    scaler = joblib.load('/all/DATA_PROCEING/classify_model/scaler_train.m')
    clf = joblib.load('/all/DATA_PROCEING/classify_model/classify_train.m')
    _, data, label, _ = read_data(all_slices, num_=type)
    data = scaler.transform(data)
    pred = clf.predict(data)
    acc = np.sum(pred == label) * 1.0 / len(data)
    print acc






if __name__ =='__main__':

    type = 116
    print type
    brain = 'edema'

    feature_dir = '/all/DATA_PROCEING/preprocessed/{}_feature_{}'.format(brain, str(type)) # 67, 83, 127,116
    post_dir = '/all/DATA_PROCEING/preprocessed/{}_post_{}/'.format(brain, str(type))
    all_slices = glob(feature_dir + '/*/*.npy')  #+ glob(post_dir + '/*/*.npy')
    print len(all_slices)
    # iter_clas(all_slices,clf=GB(), m=100, num_=type, split=80)
    five_fold(all_slices)

    # save_model(all_slices)
    # pred_with_model(post_dir)

    # slice_list, data, label, area = read_data(
    #                                  glob('/all/DATA_PROCEING/preprocessed/edema_feature' +'/*/*'))  # 430 来自segmentation_train

    # param_test1 = {'max_depth': range(1,13,1), 'n_estimators': range(20,200,10)}
    # find_para(data, label, clf=GB(), parameters=param_test1)


