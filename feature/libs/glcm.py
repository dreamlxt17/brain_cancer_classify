# coding=utf8
# 灰度共生矩阵

from skimage import io, color, feature
from skimage.filters import rank
import numpy as np
from PIL import ImageEnhance
import json
from collections import OrderedDict

def extend(s,l=40):
    while len(s) < l:
        s = s+ ' '
    return s

class GLCM(object):

    def __init__(self,
            image_path='../../data/images/1.jpg',
            # distances=[1,2,4,8],
            # distances=[1,2],
            # distances=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            # distances=[1],
            # distances = [1, 3, 5, 7, 9],
            distances = [2,4,6,8],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            threshold=None,
            level_min=1,
            level_max=256
            ):
        self.image_path = image_path
        self.distances = distances
        self.angles = angles

        # 设置灰度为 1~256
        self.n_level = level_max - level_min + 1
        self.level_min = level_min
        self.level_max = level_max

        self.glcm = self.compute_glcm()
        self.features = OrderedDict()
        self.features_mean = []
        self.features_range = []
        for i_distance in range(self.glcm.shape[2]):
            features_array = []
            for i_angle in range(self.glcm.shape[3]):
                features = self._calc_features(i_distance,i_angle)
                self.features.update(features)
                features_array.append(features.values())
            features_array = np.array(features_array)
            features_mean =  features_array.mean(0)
            features_range =  features_array.max(0) - features_array.min(0)
            self.features_mean.extend(list(features_mean))
            self.features_range.extend(list(features_range))
            self.features_all = np.array(np.array(self.features_mean) + np.array(self.features_range))
        # print len(self.features_all)

        for f in self.features.values():
            if np.fabs(f) > 10000000000:
                raise NameError('NaN 错误')

    def _calc_features(self, i_distance, i_angle):
        prefix = str(i_distance)+'_'
        p = self.glcm[:,:,i_distance,i_angle]
        p_sum = p.sum()
        p = p/p.sum()
        # 1~256
        I, J = np.ogrid[self.level_min:self.level_max+1,
                        self.level_min:self.level_max+1]
        # 2~512
        I_add = np.ogrid[2*self.level_min:2*self.level_max+1]
        # 0~255
        I_sub = np.ogrid[0:self.level_max - self.level_min + 1]

        # 变量定义
        # P_x
        p_x = p.sum(1)
        # P_y
        p_y = p.sum(0)
        # u
        u = (p*I).sum()
        # u_x
        u_x = (p_x*J[0]).mean()
        # u_y
        u_y = (p_y*J[0]).mean()
        # sigma_x
        sigma_x = max((p_x*p_sum).std(),0.00001)
        # sigma_y
        sigma_y = max((p_y*p_sum).std(),0.00001)
        # P_{x+y}
        p_add = np.zeros(2*self.n_level-1)
        # P_{x-y}
        p_sub = np.zeros(self.n_level)
        for i in range(self.n_level):
            for j in range(self.n_level):
                p_add[i+j] += p[i,j]
                p_sub[np.abs(i-j)] += p[i,j]
        # q
        '''
        q = np.zeros([self.n_level,self.n_level])
        for i in range(self.n_level):
            for j in range(self.n_level):
                denominator = p_y*p_x[i]
                numerator = p[i]*p[j]
                q[i,j] = (numerator[denominator>0]/denominator[denominator>0]).sum()
        # '''

        # 与原文区别: 
        #   原文从1计数
        #   code从0计数
        #   注意却别,尤其是乘法相关操作
        # 注意大小写I,J 
        features = OrderedDict()
        features[prefix+'angular_second_moment'] = (p**2).sum()                                        # 1, 正确
        features[prefix+'contrast'] = (p*(I-J)**2).sum()                                               # 2, 正确
        # features[prefix+'dissimilarity'] = (p*np.abs(I-J)).sum()                                       #    正确, 和2很像
        features[prefix+'correlation'] = ((I*J*p).sum() - u_x*u_y) / (sigma_x*sigma_y)                 # 3, 
        features[prefix+'variance'] = ((I - u)**2*p).sum()                                             # 4, 正确
        features[prefix+'inverse_difference_moment'] = (p/(1+(I-J)**2)).sum()                          # 5, 正确
        # features[prefix+'homogeneity'] = (p/(1+np.abs(I-J))).sum()                                     #    正确, 和5很像
        features[prefix+'sum_average'] = (I_add*p_add).sum()                                           # 6, 正确
        # features[prefix+'sum_average_other'] = (p/(1+(I-J)**2)).sum()                                  #    与原文不同
        features[prefix+'sum_entropy'] = -(p_add[p_add>0]*np.log(p_add[p_add>0])).sum()                # 8, 正确
        # 数值极大
        features[prefix+'sum_variance'] = (((I_add- features[prefix+'sum_entropy'])**2)*p_add).sum()           # 7, 数值极大,可能有错误
        # features[prefix+'sum_variance_others_1'] = (((I_add-(I_add*p_add).sum())**2)*p_add).sum()      #    正确, 和7很像
        # features[prefix+'sum_variance_others_2'] = p_add.var()                                         #    正确, 和7很像
        features[prefix+'entropy'] = -(p[p>0]*np.log(p[p>0])).sum()                                    # 9, 正确
        features[prefix+'different_variance'] = np.var(p_sub)                                          # 10,正确
        # features[prefix+'different_variance_others'] = (((I_sub -(I_sub*p_sub).sum())**2)*p_sub).sum() #    正确, 和10很像
        features[prefix+'different_entropy'] = -(p_sub[p_sub>0]*np.log(p_sub[p_sub>0])).sum()          # 11,正确

        '''
        HXY = features['entropy']
        p_x_n = np.array([[p_x[i]] for i in range(len(p_x))])
        p_y_n = np.array([p_y])
        HXY1 = -(p*np.log(p_x_n*p_y_n)).sum()
        HXY2 = -(p_x_n*p_y_n*np.log(p_x_n*p_y_n)).sum()
        # features['information_measures_of_correlation_12'] = (HXY - HXY1)
        # features['information_measures_of_correlation_13'] =
        '''
        # eigenvalue_q,_ = np.linalg.eig(q)
        # features[prefix+'maximal_correlation_coefficient'] = np.sqrt(sorted(eigenvalue_q)[-2])         # 14,正确
        # features[prefix+'maximum_probability'] = p.max()                                               # 正确
        # features[prefix+'maximum_probability_others'] = p_x.max()                                               # 正确
        return features

    
    def print_features(self):
        for i in range(len(self.features_mean)):
            print self.features_mean[i]
        for k,v in self.features.items():
            print extend(k),v
        print len(self.features_all)

    def get_features(self):
        return self.features_all

    def compute_glcm(self):
        if type(self.image_path) in [str,unicode]:
            grayImg = io.imread(self.image_path)
            print grayImg.shape
            if len(grayImg.shape) == 3:
                grayImg = color.rgb2gray(grayImg)
        else:
            grayImg = self.image_path

        # 归一化
        grayImg = grayImg.astype(np.float64)
        grayImg = (grayImg - grayImg.min())/(grayImg.max() - grayImg.min())*self.n_level
        grayImg = grayImg.astype(np.uint8)
        glcm = feature.greycomatrix(grayImg, self.distances,self.angles).astype(np.float64)
        return glcm


def main():
    GLCM().print_features()




if __name__ == '__main__':
    main()
