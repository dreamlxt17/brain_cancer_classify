# coding=utf8
from libs.glcm import GLCM
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
import os

def load_data():
    x = np.load('/all/mnist/x.train.npy')
    x = x.reshape([-1,28,28])
    y = np.load('/all/mnist/y.train.npy')
    return x,y

def classify_mnist():
    f = []
    y = []
    for i in range(500):
        if os.path.exists('../data/glcm/{:d}_y.npy'.format(i)):
            fi = np.load('../data/glcm/{:d}_f.npy'.format(i))
            yi = np.load('../data/glcm/{:d}_y.npy'.format(i))
            f.append(fi)
            y.append(yi)
    f = np.concatenate(f,0)
    y = np.concatenate(y,0)
    # f,y = load_data()
    f_len = len(f)
    f_len = 2000
    train_len = int(f_len*0.9)
    f = f[:f_len].reshape([f_len,-1])
    y = y[:f_len]
    useful_list = [0,1,2,3,5,6,7,8,9,13,15,16,17,18]
    useful_list = [2,6,9,13,16]
    index_list = []
    for ind in useful_list:
        index_list.append(ind)
    f = f[:,index_list]
    clf = svm.SVC()
    clf.fit(f[:train_len],y[:train_len])
    result = clf.predict(f[:train_len])
    print np.sum(result == y[:train_len])/float(train_len)
    result = clf.predict(f[train_len:])
    print np.sum(result == y[train_len:])/float(f_len-train_len)
    # print train_len

def classify_scene():
    f_train = []
    y_train = []
    f_test = []
    y_test = []
    for c,folder in enumerate(['MITforest','industrial','store','highway']):
        folder = os.path.join('../data/scene',folder)
        if not os.path.exists(folder):
            continue
        c_len = len(os.listdir(folder))
        for i,npy in enumerate(os.listdir(folder)):
            fi = np.load(os.path.join(folder,npy))
            index_list=[]
            useful_list = [1,2,3,5,6,7,8,9,13,15,16]
            useful_list = [5,6,8,13,16]
            for ind in useful_list:
                index_list.append(ind)
                # index_list.append(ind+19)
                # index_list.append(ind+38)
                # index_list.append(ind+57)
            # index_list=[18]
            fi = fi[index_list]
            if i < 0.9*c_len:
                f_train.append(fi)
                y_train.append(c)
            else:
                f_test.append(fi)
                y_test.append(c)
    f_train = np.array(f_train)
    # f_train = f_train/f_train.max(0)
    y_train = np.array(y_train)
    f_test = np.array(f_test)
    # f_test = f_test/f_train.max(0)
    y_test = np.array(y_test)

    print 'train',len(f_train)
    print 'test',len(f_test)
    clf = svm.SVC()
    clf.fit(f_train,y_train)
    result = clf.predict(f_train)
    print np.sum(result == y_train)/float(len(f_train))
    result = clf.predict(f_test)
    print np.sum(result == y_test)/float(len(f_test))
    # print result


def generate_features(src_dir='/all/scene/',obj_dir='../data/scene'):
    if not os.path.exists(obj_dir):
        os.mkdir(obj_dir)
    for folder in os.listdir(src_dir):
        if folder not in ['MITforest','industrial','store','highway']:
            continue
        if not os.path.exists(os.path.join(obj_dir,folder)):
            os.mkdir(os.path.join(obj_dir,folder))
        for jpg in sorted(os.listdir(os.path.join(src_dir,folder))):
            print folder,jpg
            g = GLCM(os.path.join(src_dir,folder,jpg))
            f = g.get_features()
            np.save(os.path.join(obj_dir,folder,jpg.replace('jpg','npy')),f)


def main():
    pass
    # plt.imshow(x[0])
    # plt.show()
    # GLCM(x[0]).print_features()
    # g = GLCM('../data/images/1.jpg')
    # g.print_features()
    # print g.get_features().shape
    # n = range(10)



if __name__ == '__main__':
    # main()
    classify_mnist()
    # generate_features()
    # classify_scene()
