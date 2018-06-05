# coding=utf-8
import numpy as np
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn import svm, tree
from sklearn.metrics import roc_curve, auc, classification_report
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

def split_fold(len_data, n=5):
    idxs = range(len_data)
    shuffle(idxs)
    split = len(idxs)/n
    idxs1 = idxs[:split]
    idxs2 = idxs[split:2*split]
    idxs3 = idxs[2*split:3*split]
    idxs4 = idxs[3*split:4*split]
    idxs5 = idxs[4*split:]

    tr1, te1 = idxs2+idxs3+idxs4+idxs5, idxs1
    tr2, te2 = idxs1+idxs3+idxs4+idxs5, idxs2
    tr3, te3 = idxs2+idxs1+idxs4+idxs5, idxs3
    tr4, te4 = idxs2+idxs3+idxs1+idxs5, idxs4
    tr5, te5 = idxs2+idxs3+idxs4+idxs1, idxs5

    return [[tr1, te1], [tr2, te2], [tr3, te3], [tr4, te4], [tr5, te5],]



# k-近邻
def KNN():
    '''average accuracy: 	0.852104736794 	0.749996165938'''
    clf = neighbors.KNeighborsClassifier(leaf_size=20)
    return clf

# logistic regression
def LR():
    '''average accuracy: 	0.876857704767 	0.795767195767'''
    clf = LogisticRegression(C=1.21, solver='lbfgs')
    return clf

# 朴素贝叶斯
def NB():
    '''average accuracy: 	0.645113192108 	0.643600950847'''
    clf = BernoulliNB(alpha=2, binarize=8.6)
    return clf

#决策树
def DTC():
    '''average accuracy: 	1.0 	0.792124837052'''
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    return clf

# 线性鉴别器
def LDA():
    '''average accuracy: 	0.94306850862 	0.741285177517'''
    clf = LinearDiscriminantAnalysis(solver='svd')
    return clf

# SVM  c越大越容易过拟合
def SV_M():
    '''average accuracy: 	0.948636957427 	0.789594356261'''
    clf = svm.SVC(C=3.1, gamma=0.01)
    clf= svm.SVC()
    return clf

# GBDT
def GB():
    '''average accuracy: 	1.0 	0.869994632313
    原参数　n_estimators=140,不加最后７个特征
    max_depth=15, , min_samples_split=200, n_estimators=170
    '''
    # clf = GradientBoostingClassifier(n_estimators=180, max_depth=4, min_samples_split=22)
    clf = GradientBoostingClassifier(n_estimators=90, max_depth=7)
    # clf = GradientBoostingClassifier()
    return clf

# 随机森林决策树
def RF():
    '''average accuracy: 	0.999690402477 	0.820535235028
    原参数 n_estimiter = 40
    '''
    clf = RandomForestClassifier(n_estimators=60, criterion='entropy', max_depth=9)#, max_depth=12, min_samples_split=192)
    clf = RandomForestClassifier()
    return clf

def find_para(data, label, clf=svm.SVC(), parameters=None):
    """Grid search寻找最佳参数"""

    clf = GridSearchCV(clf, parameters, n_jobs=1)
    clf.fit(data, label)
    cv_result = pd.DataFrame.from_dict(clf.cv_results_)
    with open('cv_result.csv','w') as f:
        cv_result.to_csv(f)

    print('The parameters of the best model are: ')
    print(clf.best_params_)

    y_pred = clf.predict(data)
    print(classification_report(y_true=label, y_pred=y_pred))


if __name__ == '__main__':


    # cross_roc(data, label, KNN())

    param_test1 = {'max_depth': range(1,13,1)}
    # param_test1 = {'n_neighbors':range(1,10), 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size':range(20,40)}
    # find_para(data, label, DTC(), param_test1)

