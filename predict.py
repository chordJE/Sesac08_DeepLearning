import argparse 
import torch 
import numpy as np
from pathlib import Path 

from PIL import Image 
from torchvision import transforms 

from Models.unet import Unet
from Preprocessing.unet_preprocessing import NUM_CLASSES, CLASS_NAMES


import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False 
import cv2 

PALETTE = [
    (0,   0,   0),    # 0  배경
    (220, 80,  80),   # 1  도토리
    (80,  160, 220),  # 2  밤
    (80,  200, 120),  # 3  은행
    (230, 160, 50),   # 4  피칸
    (160, 80,  200),  # 5  호박씨
    (50,  200, 200),  # 6  마카다미아
    (230, 100, 180),  # 7  브라질너트
    (100, 130, 60),   # 8  잣
    (200, 80,  50),   # 9  호두
    (80,  100, 200),  # 10 해바라기씨
    (200, 200, 60),   # 11 밤송이
    (60,  200, 160),  # 12 아몬드
    (200, 120, 80),   # 13 피스타치오
    (130, 60,  200),  # 14 땅콩
    (80,  180, 80),   # 15 캐슈넛
]

 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, 
                        help='테스트할 이미지 경로')
    return parser.parse_args()

def predict(image_size, model, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    image_target = Image.open(image_path).convert('RGB') #채널, 너비, 높이
    inp = transform(image_target).unsqueeze(0).to(device) #배치(1), 채널, 너비, 높이

    model.eval()
    with torch.no_grad():
        output = model(inp)

    #배치(1), 채널, 너비, 높이 -> 채널, 너비, 높이(시각화)
    pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
    display_image = np.array(image_target.resize((image_size, image_size)))
    return pred_mask, display_image



def draw_segmentation(image, mask, alpha=0.5, save_path=None):
    """
    이미지 위에 세그멘테이션 마스크를 반투명 컬러 오버레이로 시각화.
    중심점(각 객체 영역의 무게중심)도 함께 표시.

    image : (H, W, 3) numpy array (RGB) 또는 PIL Image
    mask  : (H, W) numpy array, 값 = 클래스 인덱스
    alpha : 마스크 투명도 (0=투명, 1=불투명)
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    img = image.copy().astype(np.float32)
    overlay = np.zeros_like(img)

    present_classes = np.unique(mask)

    for cls_idx in present_classes:
        if cls_idx == 0:  # 배경 스킵
            continue
        color = PALETTE[cls_idx % len(PALETTE)]
        overlay[mask == cls_idx] = color

    blended = img * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 각 클래스 영역의 무게중심에 점 표시
    for cls_idx in present_classes:
        if cls_idx == 0:
            continue
        region = (mask == cls_idx).astype(np.uint8)
        M = cv2.moments(region)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        color = PALETTE[cls_idx % len(PALETTE)]
        cv2.circle(blended, (cx, cy), radius=8,  color=color,         thickness=-1)
        cv2.circle(blended, (cx, cy), radius=10, color=(255,255,255), thickness=2)

    # 범례 생성
    legend_patches = [
        mpatches.Patch(color=[c / 255 for c in PALETTE[i]], label=CLASS_NAMES[i])
        for i in sorted(present_classes)
        if i != 0
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')

    cmap = plt.colormaps['tab20'].resampled(len(CLASS_NAMES))
    axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1)
    axes[1].set_title('세그멘테이션 마스크')
    axes[1].axis('off')

    axes[2].imshow(blended)
    axes[2].set_title('오버레이 + 중심점')
    axes[2].axis('off')
    if legend_patches:
        axes[2].legend(handles=legend_patches, loc='lower right',
                       fontsize=8, framealpha=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'저장 완료: {save_path}')

    plt.show()


#python predict.py --img Data/nuts/images/798592_594.jpg
if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Unet(in_channel=3, num_classes=NUM_CLASSES, base_ch=64)
    model.load_state_dict(torch.load('./unet_nuts.pth', map_location=device))
    model.to(device=device)

    pred_mask, display_image = predict(512, model, device, args.img)


    #시각화
    detected = np.unique(pred_mask)
    detected = detected[detected != 0]

    draw_segmentation(image = display_image, mask=pred_mask, alpha=0.9, save_path='./result.jpg')
    