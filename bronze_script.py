#!/usr/bin/python3

import os
import re
import cv2
import itertools
import numpy as np
import pandas as pd
from math import log
from random import random
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from sklearn.svm import LinearSVC, SVC
from nltk import FreqDist
from skimage.transform import AffineTransform, warp
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from sklearn.dummy import DummyClassifier
from skimage.transform import pyramid_gaussian
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from skimage import color, io, img_as_ubyte, img_as_float, exposure, filters
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern, corner_harris

# change the paths based on your directory
inDir1 = '/Users/zhaomengxuan/CV/bronze/禾'
inDir2 = '/Users/zhaomengxuan/CV/bronze/未'
inDir3 = '/Users/zhaomengxuan/CV/bronze/木'
outDir = '/Users/zhaomengxuan/CV/bronze/denoised'

#######################
##### import data #####
#######################

files1 = sorted([f for f in os.listdir(inDir1) if re.match(r'(?i).+\.jpg$', f)])
files2 = sorted([f for f in os.listdir(inDir2) if re.match(r'(?i).+\.jpg$', f)])
files3 = sorted([f for f in os.listdir(inDir3) if re.match(r'(?i).+\.jpg$', f)])
fullpath1 = [os.path.join(inDir1, f) for f in files1]
fullpath2 = [os.path.join(inDir2, f) for f in files2]
fullpath3 = [os.path.join(inDir3, f) for f in files3]
data1 = [color.rgb2gray(io.imread(fp1)) for fp1 in fullpath1]
data2 = [color.rgb2gray(io.imread(fp2)) for fp2 in fullpath2]
data3 = [color.rgb2gray(io.imread(fp3)) for fp3 in fullpath3]
#no_ext = [f[:-4] for f in files]
#no_ext[:5]
data1 = [img_as_float(1-image) for image in data1]
data2 = [img_as_float(1-image) for image in data2]
data3 = [img_as_float(1-image) for image in data3]
data = data1 + data2 + data3

split1, split2 = len(data1), len(data1)+len(data2)

#######################
####### padding #######
#######################

def split(n):
    if n%2 == 0:
        return int(n/2), int(n/2)
    return n//2, n//2+1

def padding(dataset):
    for image in dataset:
        if image.shape == (75,75):
            yield image
        else:
            v = 75 - image.shape[0]
            h = 75 - image.shape[1]
            top, bottom = split(v)
            left, right = split(h)
            yield cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)

data_fullsize = tuple(padding(data))
data1, data2, data3 = data_fullsize[:split1], data_fullsize[split1:split2], data_fullsize[split2:]


#######################
### artificial data ###
#######################

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    #print(shape)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    #print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def affine(dataset,scale, rotation, shear):
    assert len(dataset)==len(scale)==len(rotation)==len(shear)
    for i in range(len(dataset)):
        trans = AffineTransform(scale=scale[i], rotation=rotation[i], shear=shear[i])
        yield 1-(warp(1-dataset[i], trans))

# parameters
# scale = (1~1.2,1~1.2), rotation: -0.2~0.2, shear: -0.2~0.2, translation
scale = np.random.rand(len(data2),2)/5+1
rotation = (np.random.rand(len(data2))*2-1)/5
shear = (np.random.rand(len(data2))*2-1)/5

art_2_elastic = np.array([elastic_transform(image, alpha = 34, sigma = 4) for image in data2])
art_2_elastic[art_2_elastic>1] = 1
art_2_affine = tuple(affine(data2, scale, rotation,shear))
data = list(data1) + list(data2) + list(art_2_elastic) + list(art_2_affine) + list(data3)

print('artificial data generated')
print('='*10)
print('number of group 1:', len(fullpath1))
print('number of group 2:', len(fullpath2)*3)
print('number of group 3:', len(fullpath3))

#######################
####### denoise #######
#######################

def gamma_correction(dataset, gamma):
    for image in dataset:
        yield image ** gamma

def otsu_threshold(dataset):
    for i in range(len(dataset)):
        image = img_as_ubyte(dataset[i])
        otsu = filters.threshold_otsu(image)
        image[image>otsu] = 255
        yield img_as_float(image)

def sauvola_thre(dataset, win_size):
    for image in dataset:
        thr = filters.threshold_sauvola(image, window_size=win_size)
        yield (image > thr).astype(float)

def median_filter(dataset, kernal_size):
    for image in dataset:
        padded_image = np.pad(image, kernal_size//2, 'reflect')
        new_image = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_image[i,j] = np.median(padded_image[i:i+kernal_size, j:j+kernal_size])
        yield new_image

def alpha_trimmed_filter(dataset, size, d):
    for image in dataset:
        padded_image = np.pad(image, size//2, 'reflect')
        new_image = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                lst = sorted(padded_image[i:i+size, j:j+size].reshape(size**2))
                lst = lst[d:][:-d]
                new_image[i,j] = sum(lst)/(size**2-d*2)
        yield new_image

def combine(layer1, layer2, alpha):
    return layer1*alpha + layer2*(1-alpha)

def combine_main(dataset1, dataset2, alpha):
    assert len(dataset1) == len(dataset2)
    for i in range(len(dataset1)):
        yield combine(dataset1[i], dataset2[i], alpha)

def denoise(dataset, denoised_n_round, gamma, median_win_size, at_win_size, at_alpha, cb_alpha1, sauvola_win_size, cb_alpha2):
	data_orig = dataset
	for r in range(denoised_n_round):
		data_gamma = tuple(gamma_correction(data_orig, gamma))
		data_thr = tuple(sauvola_thre(data_gamma, sauvola_win_size))
		data_hist_eq = [exposure.equalize_hist(data_thr[i]) for i in range(len(data_thr))]
		data_denoised1 = tuple(median_filter(data_hist_eq, median_win_size))
		data_denoised2 = tuple(alpha_trimmed_filter(data_hist_eq, at_win_size,at_alpha))
		data_combine = tuple(combine_main(data_denoised1, data_denoised2, cb_alpha1))
		data_orig = tuple(combine_main(data_combine, data_orig, cb_alpha2))
	data_orig = tuple(otsu_threshold(data_orig))
	return data_orig

# parameters
dataset = data
denoised_n_round = 5
gamma = 0.3
median_win_size = 3
at_win_size = 5
at_alpha = 5
cb_alpha1 = 0.5
sauvola_win_size = 75
cb_alpha2 = 0.3

data_dn = denoise(dataset, denoised_n_round, gamma, median_win_size, at_win_size, at_alpha, cb_alpha1, sauvola_win_size, cb_alpha2)

def output(dataset, outDir, outname):
	if not os.path.exists(outDir):
		try:
			os.mkdir(outDir)
		except OSError:
			print ("Creation of the directory %s failed" % outDir)
			quit()
		else:
			print ("Successfully created the directory %s " % outDir)
	else: 
		print('Directory exists')
	for i in range(len(dataset)):
		out_name = outname + str(i).zfill(3) + '.jpg'
		out_fullpath = os.path.join(outDir, out_name)
		io.imsave(out_fullpath, dataset[i])

output(data_dn, outDir, 'denoised')
print('finished denoising, find result %s' % outDir)

#######################
####### feature #######
#######################

# GSCM
# angle = [0, np.pi/4, np.pi/2, np.pi*3/4]

def gscm(dataset, angle, distance=1):
    for image in dataset:
        image = img_as_ubyte(image)
        yield greycomatrix(image, [distance], [angle])/(image.shape[0]*image.shape[1])

def gscm_main(dataset):
    result = []
    for gscm_matrix in dataset:
        temp = {}
        
        ent = -np.sum(gscm_matrix*np.log(gscm_matrix+0.01))
        temp['entropy'] = ent
        
        amax = np.amax(gscm_matrix)
        temp['max'] = amax
        
        range_gray = 0
        for i in range(gscm_matrix.shape[0]):
            for j in range(gscm_matrix.shape[1]):
                range_gray += gscm_matrix[i,j]/(1+abs(i-j))
        temp['range_of_gray'] = range_gray[0,0]
        
        corr = greycoprops(gscm_matrix, 'correlation')
        temp['correlation'] = corr[0,0]
        
        contrast = greycoprops(gscm_matrix, 'contrast')
        temp['contrast'] = contrast[0,0]
        
        energy = greycoprops(gscm_matrix, 'energy')
        temp['energy'] = energy[0,0]
        
        asm = greycoprops(gscm_matrix, 'ASM')
        temp['ASM'] = asm[0,0]
        
        result.append(temp)
    return result

data_0_1 = tuple(gscm(data_dn, 0, distance=1))
data_45_1 = tuple(gscm(data_dn, np.pi/4, distance=1))
data_90_1 = tuple(gscm(data_dn, np.pi/2, distance=1))
data_135_1 = tuple(gscm(data_dn, np.pi*3/4, distance=1))

data_0_1_result = gscm_main(data_0_1)
data_45_1_result = gscm_main(data_45_1)
data_90_1_result = gscm_main(data_90_1)
data_135_1_result = gscm_main(data_135_1)

gscm_features = []
for i in range(len(data_dn)):
    result = list(data_0_1_result[i].values()) +\
    list(data_45_1_result[i].values()) +\
    list(data_90_1_result[i].values()) +\
    list(data_135_1_result[i].values())
    gscm_features.append(result)

print('finished generating: gray scale co-occurence matrix features')

# LBP
# local_binary_pattern(image, P, R, method='default')
def lbp(dataset, n_direction, r):
    for image in dataset:
        yield local_binary_pattern(image, n_direction, r)

lbp3 = tuple(lbp(data_dn, 8, 1))
lbp9 = tuple(lbp(data_dn, 8, 4))

lbp3_hist = [np.histogram(lbp3[i], bins=range(256))[0] for i in range(len(lbp3))]
lbp9_hist = [np.histogram(lbp3[i], bins=range(256))[0] for i in range(len(lbp9))]

print('finished generating: local binary pattern features')

# Area
def area(dataset):
    for image in dataset:
        unique, counts = np.unique(image, return_counts=True)
        yield 75**2 - dict(zip(unique, counts))[1.0]

feat_area = tuple(area(data_dn))

# Laplacian pyramid
def laplacian_prmd(dataset):
    for image in dataset:
        small1 = cv2.pyrDown(image)
        small2 = cv2.pyrDown(small1)
        large1 = cv2.pyrUp(small1)[:75,:75]
        large2 = cv2.pyrUp(small2)
        lap1 = image-large1
        lap2 = small1-large2
        yield small1, small2, lap1, lap2

laplacian_pyramid = tuple(laplacian_prmd(data_dn))

lap_small1 = [laplacian_pyramid[i][0] for i in range(len(laplacian_pyramid))]
lap_small2 = [laplacian_pyramid[i][1] for i in range(len(laplacian_pyramid))]
lap_small3 = [laplacian_pyramid[i][2] for i in range(len(laplacian_pyramid))]
lap_small4 = [laplacian_pyramid[i][3] for i in range(len(laplacian_pyramid))]

def posNneg_distribution(dataset):
    for image in dataset:
        unique, counts = np.unique(image, return_counts=True)
        dic = dict(zip(unique, counts))
        pos = sum([dic[k] for k in dic.keys() if k > 0])
        neg = sum([dic[k] for k in dic.keys() if k < 0])
        zero = sum([dic[k] for k in dic.keys() if k == 0])
        yield [pos, neg, zero]

lap_small3_distribution_feat = tuple(posNneg_distribution(lap_small3))
lap_small4_distribution_feat = tuple(posNneg_distribution(lap_small4))

def all_negative(matrix):
    return np.sum(matrix < 0)

def all_positive(matrix):
    return np.sum(matrix > 0)

lap1_feat = [[all_negative(laplacian_pyramid[i][2])]+[all_positive(laplacian_pyramid[i][2])] 
             for i in range(len(data_dn))]
lap2_feat = [[all_negative(laplacian_pyramid[i][3])]+[all_positive(laplacian_pyramid[i][3])] 
             for i in range(len(data_dn))]

def neg_per_line(matrix):
    result = []
    for i in range(matrix.shape[0]):
        ct = all_negative(matrix[i])
        result.append(ct)
    return result

def pos_per_line(matrix):
    result = []
    for i in range(matrix.shape[0]):
        ct = all_positive(matrix[i])
        result.append(ct)
    return result

def feat_by_line(image):
    nh = neg_per_line(image)
    ph = pos_per_line(image)
    nv = neg_per_line(image.T)
    pv = pos_per_line(image.T)
    return nh, ph, nv, pv

def feat_posNneg_main(dataset):
    for image in dataset:
        result = []
        # number of non-zero values in each cols/rows
        nh, ph, nv, pv = feat_by_line(image)
        # calculate features
        # number of cols/rows which contain non-zero values
        nhn0, phn0, nvn0, pvn0 = np.count_nonzero(nh),\
        np.count_nonzero(ph), np.count_nonzero(nv), np.count_nonzero(pv)
        # mean
        result.append(np.sum(nh)/nhn0)
        result.append(np.sum(ph)/phn0)
        result.append(np.sum(nv)/nvn0)
        result.append(np.sum(pv)/pvn0)
        # standard deviation
        result.append(np.std(nh))
        result.append(np.std(ph))
        result.append(np.std(nv))
        result.append(np.std(pv))
        # smoothess: 1 - (1/(1+mean**2))
        result.append(1 - (1/(1+(np.sum(nh)/nhn0)**2)))
        result.append(1 - (1/(1+(np.sum(ph)/phn0)**2)))
        result.append(1 - (1/(1+(np.sum(nv)/nvn0)**2)))
        result.append(1 - (1/(1+(np.sum(pv)/pvn0)**2)))
        # entropy
        Pnh, Pph, Pnv, Ppv = nh/sum(nh), ph/sum(ph), nv/sum(nv), pv/sum(pv)
        result.append(np.sum([-(Pnh[i]*log(Pnh[i]+0.01)) for i in range(len(Pnh))]))
        result.append(np.sum([-(Pph[i]*log(Pph[i]+0.01)) for i in range(len(Pph))]))
        result.append(np.sum([-(Pnv[i]*log(Pnv[i]+0.01)) for i in range(len(Pnv))]))
        result.append(np.sum([-(Ppv[i]*log(Ppv[i]+0.01)) for i in range(len(Ppv))]))
        yield result

lap_large = [laplacian_pyramid[i][2] for i in range(len(laplacian_pyramid))]
lap_small = [laplacian_pyramid[i][3] for i in range(len(laplacian_pyramid))]

lap_large_feat = tuple(feat_posNneg_main(lap_large))
lap_small_feat = tuple(feat_posNneg_main(lap_small))

# Sobel filter 3*3
def sobelH(dataset):
    for image in dataset:
        yield filters.sobel_h(image)

def sobelV(dataset):
    for image in dataset:
        yield filters.sobel_v(image)

def sobelSqrt(dataset):
    for image in dataset:
        yield filters.sobel(image)

sb_h, sb_v, sb_sqrt = tuple(sobelH(data_dn)), tuple(sobelV(data_dn)), tuple(sobelSqrt(data_dn))

sobel_sum = [np.count_nonzero(image) for image in sb_sqrt]
sobel_h_distribution = tuple(posNneg_distribution(sb_h))
sobel_v_distribution = tuple(posNneg_distribution(sb_v))

# Roberts filter (diagonal)
def robertsP(dataset):
    for image in dataset:
        yield (filters.roberts_pos_diag(image))

def robertsN(dataset):
    for image in dataset:
        yield (filters.roberts_neg_diag(image))

def robertsPSqrt(dataset):
    for image in dataset:
        yield (filters.roberts(image))

rt_p, rt_n, rt_sqrt = tuple(robertsP(data_dn)), tuple(robertsN(data_dn)), tuple(robertsPSqrt(data_dn))

rt_sum = [np.count_nonzero(image) for image in rt_sqrt]
rt_p_distribution = tuple(posNneg_distribution(rt_p))
rt_n_distribution = tuple(posNneg_distribution(rt_n))

# Laplace filter
def _mask_filter_result(result):
    """Return result after masking.
    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    result[0, :] = 0
    result[-1, :] = 0
    result[:, 0] = 0
    result[:, -1] = 0
    return result

def Lap(image, l_matrix):
    image = img_as_float(image)
    result = convolve(image, l_matrix)
    result = _mask_filter_result(result)
    b,s = 1/np.amax(result), 1/abs(np.amin(result))
    result[result<0] *= s
    result[result>0] *= b
    return result

def lap_main(dataset,l_matrix):
    for image in dataset:
        yield (Lap(image, l_matrix))

l = np.array([[0,0,-1,0,0], [0,-1,-2,-1,0], [-1,-2,16,-2,-1], [0,-1,-2,-1,0], [0,0,-1,0,0]])

lap = tuple(lap_main(data_dn, l))

# corner detection
def harris(dataset):
    for image in dataset:
        feat = corner_harris(image, k=0.1)
        feat[np.abs(feat)<2] = 0
        yield feat[3:72, 3:72]

corner = tuple(harris(data_dn))
harris_distribution_feat = tuple(posNneg_distribution(corner))
ncn = [all_negative(corner[i]) for i in range(len(corner))]
pcn = [all_positive(corner[i]) for i in range(len(corner))]
corner_feat = tuple(feat_posNneg_main(corner))

# Filter banks
h0 = np.array([0.5, 0.5])
h1 = np.array([0.5, -0.5])

def filter_banks(image):
    L = np.array([np.convolve(image[i], h0) for i in range(image.shape[0])][::2])
    H = np.array([np.convolve(image[i], h1) for i in range(image.shape[0])][::2])
    LL = np.array([np.convolve(L.T[i], h0) for i in range(image.shape[1])][::2]).T
    LH = np.array([np.convolve(L.T[i], h1) for i in range(image.shape[1])][::2]).T
    HL = np.array([np.convolve(H.T[i], h0) for i in range(image.shape[1])][::2]).T
    HH = np.array([np.convolve(H.T[i], h1) for i in range(image.shape[1])][::2]).T
    return LL, LH, HL, HH

LL, LH, HL, HH = [], [], [], []
for image in data_dn:
    ll, lh, hl, hh = filter_banks(image)
    LL.append(ll)
    LH.append(lh)
    HL.append(hl)
    HH.append(hh)

HL_distribution_feat = tuple(posNneg_distribution(HL))
LH_distribution_feat = tuple(posNneg_distribution(LH))
HH_distribution_feat = tuple(posNneg_distribution(HH))

# horizontal and vertical info
def pixels_per_line(matrix):
    result = []
    for i in range(matrix.shape[0]):
        ct = np.count_nonzero(1-matrix[i])
        result.append(ct)
    return result

def pixel_count(image):
    h = pixels_per_line(image)
    v = pixels_per_line(image.T)
    return h,v

def std(array):
    array = list(filter(lambda a: a != 0, array))
    return np.std(array)

def greater_than_half(array):
    assert(np.amax(array) == 1 and np.amin(array) >= 0)
    array = list(filter(lambda a: a > 0.5, array))
    return len(array)

def less_than_half(array):
    assert(np.amax(array) == 1 and np.amin(array) >= 0)
    array = list(filter(lambda a: a < 0.5 and a > 0, array))
    return len(array)

def ent_by_line(image):
    result = []
    for i in range(image.shape[0]):
        rev = img_as_ubyte(1-image)
        probDic = FreqDist(rev[i])
        probDic = {k:v/len(image[i]) for k,v in probDic.items()}
        prob = [probDic[pixel] for pixel in rev[i]]
        ent = sum([-p*log(p) for p in prob])
        result.append(ent)
    return sorted(result, reverse=True)

def ent_by_line_main(image):
    ent_h = ent_by_line(image)
    ent_v = ent_by_line(image.T)
    return ent_h, ent_v

# read images of black(0) background and lighter color as character (non-zero)
# used for original image/down-sampled image

def feat_main(dataset):
    for image in dataset:
        result = []
        h,v = pixel_count(image)

        # calculate features
        hn0, vn0 = np.count_nonzero(h), np.count_nonzero(v)

        # mean
        result.append(sum(h)/hn0)
        result.append(sum(v)/vn0)

        # standard deviation
        result.append(std(h))
        result.append(std(v))

        # smoothess: 1 - (1/(1+mean**2))
        result.append(1 - (1/(1+(sum(h)/hn0)**2)))
        result.append(1 - (1/(1+(sum(v)/vn0)**2)))

        # entropy
        Ph, Pv = np.array(h)/sum(h), np.array(v)/sum(v)
        result.append(sum([-(Ph[i]*log(Ph[i]+0.01)) for i in range(len(Ph))]))
        result.append(sum([-(Pv[i]*log(Pv[i]+0.01)) for i in range(len(Pv))]))

        # count non-zero pixels
        max_h, max_v = max(h), max(v)
        result.append(max_h)
        result.append(max_v)

        # rank non-zero pixels
        sorted_h = sorted(list(np.array(h)/max_h))[::-1]
        sorted_v = sorted(list(np.array(v)/max_v))[::-1]

        # mean, std of sorted prob
        result.append(sum(sorted_h)/hn0)
        result.append(sum(sorted_v)/vn0)
        result.append(std(sorted_h))
        result.append(std(sorted_v))
        greater_ratio_h, greater_ratio_v = greater_than_half(sorted_h)/hn0, greater_than_half(sorted_v)/vn0
        result.append(greater_ratio_h)
        result.append(greater_ratio_v)
        
        # mean, std of ent_by_line
        ent_h, ent_v = ent_by_line_main(image)
        ent_hn0, ent_vn0 = np.count_nonzero(ent_h), np.count_nonzero(ent_v)
        ent_h_mean, ent_v_mean = sum(ent_h)/ent_hn0, sum(ent_v)/ent_vn0
        ent_h_std, ent_v_std = std(ent_h), std(ent_v)
        result.append(np.amax(ent_h))
        result.append(np.amax(ent_v))
        result.append(ent_h_mean)
        result.append(ent_v_mean)
        result.append(ent_h_std)
        result.append(ent_v_std)
        
        yield result

sb_sqrt_rev = [1-image for image in sb_sqrt]
rt_sqrt_rev = [1-image for image in rt_sqrt]

feat_all_data = tuple(feat_main(data_dn))
feat_lap_small1 = tuple(feat_main(lap_small1))
feat_lap_small2 = tuple(feat_main(lap_small2))

feat_sobel_h = tuple(feat_posNneg_main(sb_h))
feat_sobel_v = tuple(feat_posNneg_main(sb_v))
feat_sobel_sqrt = tuple(feat_main(sb_sqrt_rev))

feat_roberts_p = tuple(feat_posNneg_main(rt_p))
feat_roberts_n = tuple(feat_posNneg_main(rt_n))
feat_roberts_sqrt = tuple(feat_main(rt_sqrt_rev))

feat = []
for i in range(len(data_dn)):
    feat.append([])
    feat[i].append(gscm_features[i])
    feat[i].append(lbp3_hist[i])
    feat[i].append(lbp9_hist[i])
    feat[i].append(lap_large_feat[i])
    feat[i].append(lap_small_feat[i])
    feat[i].append(feat_all_data[i])
    feat[i].append(feat_lap_small1[i])
    feat[i].append(feat_lap_small2[i])
    feat[i].append(sobel_h_distribution[i])
    feat[i].append(sobel_v_distribution[i])
    feat[i].append(feat_sobel_h[i])
    feat[i].append(feat_sobel_v[i])
    feat[i].append(feat_sobel_sqrt[i])
    feat[i].append(rt_p_distribution[i])
    feat[i].append(rt_n_distribution[i])
    feat[i].append(feat_roberts_p[i])
    feat[i].append(feat_roberts_n[i])
    feat[i].append(feat_roberts_sqrt[i])
    feat[i].append([feat_area[i]])
    feat[i].append([sobel_sum[i]])
    feat[i].append([rt_sum[i]])
    feat[i].append(harris_distribution_feat[i])
    feat[i].append(lap_small3_distribution_feat[i])
    feat[i].append(lap_small4_distribution_feat[i])
    feat[i].append(HL_distribution_feat[i])
    feat[i].append(LH_distribution_feat[i])
    feat[i].append(HH_distribution_feat[i])

for i in range(len(feat)):
    feat[i] = list(itertools.chain.from_iterable(feat[i]))
feat = np.array(feat)
print(feat.shape)

# labels
label = np.concatenate((np.repeat(0,78), np.repeat(1,72), np.repeat(2,114)), axis=0)

# train/test spliting
X_test, y_test = feat[::5], label[::5]
X_train = [feat[i].astype(float) for i in range(len(feat)) if i%5 != 0]
y_train = [label[i] for i in range(len(label)) if i%5 != 0]

#######################
###### base line ######
#######################

clf_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
clf_dummy.fit(X_train, y_train)
print('Base line - dummy classifier score:', clf_dummy.score(X_test, y_test))

#######################
######   SVM1   #######
#######################
print('linear SVM with all %s features' % feat.shape[1])

clf = LinearSVC()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

matrix1 = pd.DataFrame(confusion_matrix(y_test, pred), 
             columns=['pred_1', 'pred_2', 'pred_3'], index=['class_1', 'class_2', 'class_3'])
precision1 = precision_score(y_test, pred, average='macro')
recall1 = recall_score(y_test, pred, average='macro')
f1_1 = f1_score(y_test, pred, average='macro')
score1 = clf.score(X_test, y_test)

print('='*10)
print('precision: {0:.3f}\nrecall: {0:.3f}\nf1 score: {0:.3f}'.format(precision1, recall1, f1_1))
print('classifier score: {0:.3f}'.format(score1))
print(matrix1)

#######################
## feature reduction ##
#######################
print('SelectKBest feature reduction')

fit = SelectKBest(chi2, k=41).fit(feat, label)
feat_new = fit.transform(feat)
print('='*10)
print('model improving')
print('number of features reduced to:', feat_new.shape[1])

X_test_new = feat_new[::5]
X_train_new = [feat_new[i].astype(float) for i in range(len(feat_new)) if i%5 != 0]
clf_new = LinearSVC()
clf_new.fit(X_train_new, y_train)
pred_new = clf_new.predict(X_test_new)

matrix2 = pd.DataFrame(confusion_matrix(y_test, pred_new), 
             columns=['pred_1', 'pred_2', 'pred_3'], index=['class_1', 'class_2', 'class_3'])
precision2 = precision_score(y_test, pred_new, average='macro')
recall2 = recall_score(y_test, pred_new, average='macro')
f1_2 = f1_score(y_test, pred_new, average='macro')
score2 = clf_new.score(X_test_new, y_test)

print('='*10)
print('precision: {0:.3f}\nrecall: {0:.3f}\nf1 score: {0:.3f}'.format(precision2, recall2, f1_2))
print('classifier score: {0:.3f}'.format(score2))
print(matrix2)

#######################
########  PCA  ########
#######################
print('PCA feature reduction')

pca = PCA(n_components=2)
pca_feat = pca.fit_transform(feat)
X_test_pca = pca_feat[::5]
X_train_pca = [pca_feat[i] for i in range(len(pca_feat)) if i%5 != 0]

clf_pca = LinearSVC()
clf_pca.fit(X_train_pca, y_train)

pred_pca = clf_pca.predict(X_test_pca)
matrix3 = pd.DataFrame(confusion_matrix(y_test, pred_pca), 
             columns=['pred_1', 'pred_2', 'pred_3'], index=['class_1', 'class_2', 'class_3'])
precision3 = precision_score(y_test, pred_pca, average='macro')
recall3 = recall_score(y_test, pred_pca, average='macro')
f1_3 = f1_score(y_test, pred_pca, average='macro')
score3 = clf_pca.score(X_test_pca, y_test)

print('='*10)
print('precision: {0:.3f}\nrecall: {0:.3f}\nf1 score: {0:.3f}'.format(precision3, recall3, f1_3))
print('classifier score: {0:.3f}'.format(score3))
print(matrix3)

#######################
#### Leave-one-out ####
#######################
print('running leave-one-out evaluation')

loo = LeaveOneOut()
acc = []
for train_index, test_index in loo.split(feat):
    X_train_one, X_test_one = feat[train_index], feat[test_index]
    y_train_one, y_test_one = label[train_index], label[test_index]
    
    clf_one = LinearSVC()
    clf_one.fit(X_train_one, y_train_one)
    pred_one = clf_one.predict(X_test_one)
    
    acc.append(precision_score(y_test_one, pred_one, average='macro'))

acc1 = sum(acc)/len(acc)

acc_new = []
for train_index, test_index in loo.split(feat_new):
    X_train_one_new, X_test_one_new = feat_new[train_index], feat_new[test_index]
    y_train_one, y_test_one = label[train_index], label[test_index]
    
    clf_one = LinearSVC()
    clf_one.fit(X_train_one_new, y_train_one)
    pred_one = clf_one.predict(X_test_one_new)
    
    acc_new.append(precision_score(y_test_one, pred_one, average='macro'))

acc2 = sum(acc_new)/len(acc_new)

acc_pca = []
for train_index, test_index in loo.split(pca_feat):
    X_train_one_pca, X_test_one_pca = pca_feat[train_index], pca_feat[test_index]
    y_train_one, y_test_one = label[train_index], label[test_index]
    
    clf_one = LinearSVC()
    clf_one.fit(X_train_one_pca, y_train_one)
    pred_one = clf_one.predict(X_test_one_pca)
    
    acc_pca.append(precision_score(y_test_one, pred_one, average='macro'))

acc3 = sum(acc_pca)/len(acc_pca)

print('='*10)
print('SVM accuracy: {0:.3f}\nSelectKBest accuracy: {0:.3f}\nPCA accuracy: {0:.3f}'.format(acc1, acc2, acc3))

