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


def rotate(img, boxes, angle=None):
    if angle is None:
        angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    boxes = rotate_boxes(boxes, -angle, w, h)   # cv2는 반시계, 수식은 시계 방향
    return img, clip_boxes(boxes)

def translate(img, boxes, tx=None, ty=None):
    h, w = img.shape[:2]
    if tx is None:
        tx = random.uniform(-0.1, 0.1)
    if ty is None:
        ty = random.uniform(-0.1, 0.1)
    M   = np.float32([[1, 0, tx * w], [0, 1, ty * h]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    boxes = [(cls, cx + tx, cy + ty, bw, bh) for cls, cx, cy, bw, bh in boxes]
    return img, clip_boxes(boxes)

def adjust_brightness(img, boxes, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img, boxes

def adjust_contrast(img, boxes, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    mean = img.mean()
    img  = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return img, boxes

def gaussian_blur(img, boxes):
    ksize = random.choice([3, 5, 7])
    img   = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img, boxes

def gaussian_noise(img, boxes, scale=10):
    noise = np.random.normal(0, scale, img.shape).astype(np.float32)
    img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, boxes

def clip_boxes(boxes):
    """박스가 이미지 밖으로 나가지 않도록 클리핑, 사라진 박스 제거"""
    result = []
    for cls, cx, cy, w, h in boxes:
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(1.0, cx + w / 2)
        y2 = min(1.0, cy + h / 2)
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w > 0.01 and new_h > 0.01:   # 너무 작아진 박스 제거
            result.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, new_w, new_h))
    return result


def rotate_boxes(boxes, angle_deg, img_w, img_h):
    """박스 4개 꼭짓점을 회전시켜 새 axis-aligned bbox 계산"""
    cx_img, cy_img = img_w / 2, img_h / 2
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    result = []
    for cls, cx, cy, w, h in boxes:
        # 정규화 → 픽셀
        cx_p, cy_p = cx * img_w, cy * img_h
        w_p,  h_p  = w  * img_w, h  * img_h

        # 4 꼭짓점
        corners = np.array([
            [cx_p - w_p/2, cy_p - h_p/2],
            [cx_p + w_p/2, cy_p - h_p/2],
            [cx_p + w_p/2, cy_p + h_p/2],
            [cx_p - w_p/2, cy_p + h_p/2],
        ])

        # 이미지 중심 기준 회전
        corners -= [cx_img, cy_img]
        rotated = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated += [cx_img, cy_img]

        # 새 axis-aligned bbox
        x1, y1 = rotated.min(axis=0)
        x2, y2 = rotated.max(axis=0)

        # 픽셀 → 정규화
        result.append((cls,
                        (x1 + x2) / 2 / img_w,
                        (y1 + y2) / 2 / img_h,
                        (x2 - x1) / img_w,
                        (y2 - y1) / img_h))
    return result




import albumentations as A

def play_albumentation():
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
