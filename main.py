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

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#0818 추가(Unet 관련 모듈)
from Models.unet import Unet, train
from Preprocessing.unet_preprocessing import get_dataloader, NUM_CLASSES

def count_params(model):
    #requires_grad = True (훈련시킬것, 변경가능) = Fasle(훈련안시킴, 변경불가)
    #p.numel(파라미터의 구성요소 개수)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'전체 파라미터    : {total:,}')
    print(f'훈련 가능 파라미터: {trainable:,}')
    print(f'동결된 파라미터  : {total - trainable:,}')

if __name__ == '__main__':
    image_dir = r'C:\Users\jeong\OneDrive\Desktop\Lecture\Data\NutsDataset\images'
    label_dir = r'C:\Users\jeong\OneDrive\Desktop\Lecture\Data\NutsDataset\labels'

    train_loader, valid_loader = get_dataloader(image_dir, label_dir, 
                                                image_size=512, batch_size=4)

    model = Unet(in_channel=3, num_classes=NUM_CLASSES)
    count_params(model=model)

    train(model, train_loader=train_loader, valid_loader=valid_loader,
          epochs=1, lr=1e-3, save_path ='./unet_nuts.pth', num_classes=NUM_CLASSES)








#딥러닝 시퀀스
#1.데이터 가져옴
#2.데이터 정제(preprocessing)
#3.알고리즘선택
#4.훈련
#5.검증
#6.평가
#7.배포