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

def count_params(model):
    #requires_grad = True (훈련시킬것, 변경가능) = Fasle(훈련안시킴, 변경불가)
    #p.numel(파라미터의 구성요소 개수)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'전체 파라미터    : {total:,}')
    print(f'훈련 가능 파라미터: {trainable:,}')
    print(f'동결된 파라미터  : {total - trainable:,}')

if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    #Faster RCNN
    #Backbone => 이미지의 특징 추출 => 저수준의 특징 추출(ResNet)
    #RPN(Resion Proposal Networks) => Bounding Box의 후보 제안 
    #ROI Head => RPN 를 본 뒤, 분류 수행, bbox 보정**
    #Faster RCNN : 클래스 = 분류하고자 하는 객체의 개수
    #ROI Head를 변경
    #print(model)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #ROI Head도 2개의 부속품이 있음 -> Cls_score(분류) / BBox_predictor(바운딩박스 찾기)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1+1)

    count_params(model)










#딥러닝 시퀀스
#1.데이터 가져옴
#2.데이터 정제(preprocessing)
#3.알고리즘선택
#4.훈련
#5.검증
#6.평가
#7.배포