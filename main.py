#라벨만들기 위한 os, json
import os, json, re
import matplotlib.pyplot as plt 

#비전 관련 작업 cv로 진행할 때, numpy를 함께 임포트
import cv2 
import numpy as np

import shutil #.sh / .bash

#실제 코드 augmentation.py -> 호출 main.py
from Utils import augmentation as aug
import albumentations as A

if __name__ == '__main__':
    #실행 연습
    # fig, ax = plt.subplots(1, 2)
    # #image = cv2.flip(image, 1)
    # image = r'./Data/YoloAugmentation/images/train/A220120XX_10306.jpg'
    # image = cv2.imread(image)
    # print(image.shape)
    # ax[0].imshow(image)
    # image, label = aug.flip_horizontal(image, None)
    # ax[1].imshow(image)
    # plt.show()
    # print(label)

    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    image = r'./Data/YoloAugmentation/images/train/A220120XX_10307.jpg'
            
    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    plt.imshow(transformed_image)
    plt.show()

    # aug.pipe_augmentation()









#딥러닝 시퀀스
#1.데이터 가져옴
#2.데이터 정제(preprocessing)
#3.알고리즘선택
#4.훈련
#5.검증
#6.평가
#7.배포