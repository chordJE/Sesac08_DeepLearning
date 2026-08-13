import cv2 
import numpy as np
import shutil
import random

#
from tqdm import tqdm
import matplotlib.pyplot as plt 

from pathlib import Path #경로 + glob

#가장 기본이 되는 경로(BASE_DIR)은 여기야~
BASE_DIR = Path(r'C:\Users\jeong\OneDrive\Desktop\Lecture\Data') 
SRC_DIR = BASE_DIR / 'PeachDataset' / 'YoloDataset'
DST_DIR = BASE_DIR / 'YoloAugmentation'


#1.증강된 데이터셋을 구성하기 위한 준비
def create_folder(src_folder, dst_folder):
    #1.YOLO 데이터셋에 대해 복사->증강(이미지)
    #디렉토리 있는지 확인, 복사
    # source_dir = r'./Data/PeachDataset/YoloDataset'
    # destination_dir = r'./Data/YoloAugmentation' 
    # os.path.exists(경로)
    # if not os.path.exists(destination_dir):
    #     os.mkdir(destination_dir)

    shutil.copytree(src_folder, dst_folder)
    print(f'복사 완료 : {dst_folder}')

#2.실제 증강 수행 -> 랜덤 실행! 
def augmentation_image(image, label):
    '''이미지 1장(image)에 대해서, 랜덤한 증강 조합 적용'''

    if random.random() < 0.5:
        image, label = flip_horizontal(image, label)
    if random.random() < 0.5:
        image, label = flip_vertical(image, label)
    # if random.random() < 0.5:
    #     image, label = rotate(image, label)
    # if random.random() < 0.5:
    #     image, label = translate(image, label)
    # if random.random() < 0.5:
    #     image, label = gaussian_blur(image, label)
    # if random.random() < 0.5:
    #     image, label = gaussian_noise(image, label)
    # if random.random() < 0.5:
    #     image, label = adjust_brightness(image, label)
    # if random.random() < 0.5:
    #     image, label = adjust_contrast(image, label)

    return image, label


#욜로 라벨을 분리, cls, cx, cy, w, h로 로드
def load_yolo_label(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, cx, cy, w, h))
        return boxes


def save_yolo_label(label_path, labels):
    lines = [f'{cls}    {cx:.6f}    {cy:.6f}    {w:.6f}    {h:.6f}' 
                    for cls, cx, cy, w, h in labels]

    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))


#최종 증강 파이프라인
#나의 train image, labels->모두 증강하는 반복문
def pipe_augmentation(n=3):
    image_dir = DST_DIR / 'images' / 'train'
    label_dir = DST_DIR / 'labels' / 'train'

    #glob   => python 내장 라이브러리 / 파일, 폴더 경로 컨트롤
    # *(all).jpg => 파일이름(*).jpg  => jpg로 끝나는 모든 파일
    #sorted => 정렬 / 오름차순 작->큰, a->z, ㄱ->ㅎ
    #images_files 는 images/train 안에 있는 모든 그림파일
    image_files = sorted(image_dir.glob('*.jpg'))
    print(f'증강 프로세스 시작 : {len(image_files)} 파일을 {n}배 증강')

    for image_path in tqdm(image_files, desc='Augmentation processing...'):
        filename = image_path.stem
        label_path = label_dir / f'{filename}.txt'

        #파이프라인을 한 시퀀스 돌아라!
        image = cv2.imread(str(image_path))
        label = load_yolo_label(label_path)

        #1장의 이미지-라벨 쌍에 대하여 n번의 증강
        for i in range(n):
            augmented_image, augmented_label = augmentation_image(image, label)
            out_name = f'{filename}_{i}'
            cv2.imwrite(str(image_dir/f'{out_name}.jpg'), augmented_image)
            save_yolo_label(label_dir/f'{out_name}.txt', augmented_label)
        break
    

#3. 실제 증강 함수
def flip_horizontal(image, label):
    #opencv에 설정된 이미지 뒤집기 함수(flip) -> 1(좌우) / 0(위아래) / -1(대각선)
    image = cv2.flip(image, 1)
    label = [(cls, 1.0-cx, cy, w, h) for cls, cx, cy, w, h in label]
    return image, label

def flip_vertical(image, label):
    image = cv2.flip(image, 0)
    label = [(cls, cx, 1.0-cy, w, h) for cls, cx, cy, w, h in label]
    return image, label
