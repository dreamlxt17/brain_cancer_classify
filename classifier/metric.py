# coding=utf-8
'''计算各种分类performance'''

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score, precision_recall_curve
import numpy as np

def comput_recall(label, pred):
    return recall_score(label, pred)

def comput_precision(label, pred):
    return precision_score(label, pred)

def comput_accuracy(label, pred):
    return accuracy_score(label, pred)

def comput_tpn(label, pred, tr_label=1):
    tp = 0
    for l, p in zip(label, pred):
        if l==p and l==tr_label:
            tp += 1
    return tp

def comput_fpn(label, pred, tr_label=1):
    fp = 0
    for l, p in zip(label, pred):
        if l != p and l==tr_label:
            fp += 1
    return fp

def comput_specificity(label, pred):
    FP = comput_fpn(label, pred)
    TN = comput_tpn(label, pred, tr_label=0)

    return TN*1.0/(TN+FP)

def comupt_precision(label, pred):
    pass

def comput_residual(target, prob):
    residual = np.array(target) - np.array(prob)
    return np.percentile(residual, [2.5, 97.5])




if __name__=='__main__':
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    classifier = svm.SVC()
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)

    print comput_specificity(y_test, pred)




