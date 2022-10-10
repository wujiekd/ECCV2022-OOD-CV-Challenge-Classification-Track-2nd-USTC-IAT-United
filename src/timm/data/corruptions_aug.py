'''

Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ICLR2019)
Paper link: https://arxiv.org/abs/1903.12261

Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming
Paper link: https://arxiv.org/abs/1907.07484

Modified by Xiaofeng Mao 
2021.8.30
'''

from imagecorruptions import corrupt
import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops , ImageFilter
import PIL
import numpy as np

def Corruption_Gaussian_Noise(img,level):
    corrupted_image = corrupt(img, corruption_name='gaussian_noise', severity=level)
    return corrupted_image

def Corruption_Shot_Noise(img,level):
    corrupted_image = corrupt(img, corruption_name='shot_noise', severity=level)
    return corrupted_image

def Corruption_Impulse_Noise(img,level):
    corrupted_image = corrupt(img, corruption_name='impulse_noise', severity=level)
    return corrupted_image

def Corruption_Defocus_Blur(img,level):
    corrupted_image = corrupt(img, corruption_name='defocus_blur', severity=level)
    return corrupted_image

def Corruption_Glass_Blur(img,level):
    corrupted_image = corrupt(img, corruption_name='glass_blur', severity=level)
    return corrupted_image

def Corruption_Motion_Blur(img,level):
    corrupted_image = corrupt(img, corruption_name='motion_blur', severity=level)
    return corrupted_image

def Corruption_Zoom_Blur(img,level):
    corrupted_image = corrupt(img, corruption_name='zoom_blur', severity=level)
    return corrupted_image

def Corruption_Snow(img,level):
    corrupted_image = corrupt(img, corruption_name='snow', severity=level)
    return corrupted_image

def Corruption_Frost(img,level):
    corrupted_image = corrupt(img, corruption_name='frost', severity=level)
    return corrupted_image

def Corruption_Fog(img,level):
    corrupted_image = corrupt(img, corruption_name='fog', severity=level)
    return corrupted_image

def Corruption_Brightness(img,level):
    corrupted_image = corrupt(img, corruption_name='brightness', severity=level)
    return corrupted_image

def Corruption_Contrast(img,level):
    corrupted_image = corrupt(img, corruption_name='contrast', severity=level)
    return corrupted_image

def Corruption_Elastic_Transform(img,level):
    corrupted_image = corrupt(img, corruption_name='elastic_transform', severity=level)
    return corrupted_image

def Corruption_Pixelate(img,level):
    corrupted_image = corrupt(img, corruption_name='pixelate', severity=level)
    return corrupted_image

def Corruption_Jpeg_Compression(img,level):
    corrupted_image = corrupt(img, corruption_name='jpeg_compression', severity=level)
    return corrupted_image

def Corruption_Speckle_Noise(img,level):
    corrupted_image = corrupt(img, corruption_name='speckle_noise', severity=level)
    return corrupted_image

def Corruption_Gaussian_Blur(img,level):
    corrupted_image = corrupt(img, corruption_name='gaussian_blur', severity=level)
    return corrupted_image

def Corruption_Spatter(img,level):
    corrupted_image = corrupt(img, corruption_name='spatter', severity=level)
    return corrupted_image

def Corruption_Saturate(img,level):
    corrupted_image = corrupt(img, corruption_name='saturate', severity=level)
    return corrupted_image


def Raw(img ,level):
    return img



def Corr_augment_list(m):  # 16 oeprations and their ranges
    l = [
        [
            (Raw),
            (Corruption_Gaussian_Noise),
            (Corruption_Shot_Noise),
            (Corruption_Impulse_Noise),
            (Corruption_Speckle_Noise),

            # (Corruption_Defocus_Blur),
            # (Corruption_Glass_Blur),
            # (Corruption_Motion_Blur),
            # (Corruption_Zoom_Blur),
            # (Corruption_Gaussian_Blur),

            (Corruption_Snow),
            (Corruption_Frost),
            (Corruption_Fog),
            (Corruption_Brightness),

            (Corruption_Contrast),
            (Corruption_Elastic_Transform),
            (Corruption_Pixelate),
            (Corruption_Jpeg_Compression),
            (Corruption_Spatter),
            (Corruption_Saturate),
        ],

        [
            (Raw),
            (Corruption_Snow),
            (Corruption_Frost),
            (Corruption_Fog),
            (Corruption_Brightness),

        ],

        [
            (Raw),
            (Corruption_Gaussian_Noise),
            (Corruption_Shot_Noise),
            (Corruption_Impulse_Noise),
            (Corruption_Speckle_Noise),


        ],

        [
            (Raw),
            (Corruption_Defocus_Blur),
            (Corruption_Glass_Blur),
            (Corruption_Motion_Blur),
            (Corruption_Zoom_Blur),
            (Corruption_Gaussian_Blur),

        ],

        [
            (Raw),
            (Corruption_Contrast),
            (Corruption_Elastic_Transform),
            (Corruption_Pixelate),
            (Corruption_Jpeg_Compression),
            (Corruption_Spatter),
            (Corruption_Saturate),

            (Corruption_Snow),
            (Corruption_Frost),
            (Corruption_Fog),
            (Corruption_Brightness),

            # (Corruption_Gaussian_Noise),
            # (Corruption_Shot_Noise),
            # (Corruption_Impulse_Noise),
            # (Corruption_Speckle_Noise),

        ],

    ]


    return l[m]

class Corruption_RandAugment:
    def __init__(self, n = 1  , m = 4 , max_level = 2):  # n指定选择1种,m指定第几种策略,max_level设置扰动等级
        self.n = n
        self.max_level = max_level
        self.augment_list = Corr_augment_list(m)
        self.choice_weights = [1 / len(self.augment_list) for i in range(len(self.augment_list))]
        print("choose corr num is ",str(n))
        print("max_level is ",str(max_level))
        self.opera = [i for i in range(len(self.augment_list))]

    def __call__(self, img):
        ops_num = np.random.choice(self.opera ,self.n, p=self.choice_weights)
        level = np.random.choice([i+1 for i in range(self.max_level)], 1)[0] # 1,2,3,4,5
        img = np.array(img)     
        #print(img.shape)
        
        for i in ops_num:
            op = self.augment_list[i]
            img = op(img, level)
        img=Image.fromarray(img) 

        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.n}, ops='
        for op in self.augment_list:
            fs += f'\n\t{op}'
        fs += ')'
        return fs

# def SAR_test_augment_list():  # 16 oeprations and their ranges
#     l = [
#             (Raw, 0 ),
#             (MedianFilter, 3),
#             (MedianFilter, 5),
#             (GaussianBlur, 2),
#             (EDGE_ENHANCE, 2),
#         ]

#     return l

# class SAR_test_RandAugment:
#     def __init__(self, n = 0):  #
#         self.n = n
#         self.augment_list = SAR_test_augment_list()
#         self.ops = [self.augment_list[self.n]]
#         print(self.ops)

#     def __call__(self, img):
#         for op, val in self.ops:
#             #print(op, val)
#             img = op(img, val)

#         return img

#     def __repr__(self):
#         fs = self.__class__.__name__ + f'(n={self.n}, ops='
#         for op in self.augment_list:
#             fs += f'\n\t{op}'
#         fs += ')'
#         return fs