import torch
import torchvision 
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
                                        FasterRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tqdm import tqdm

def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    #Backbone => 이미지의 특징 추출 => 저수준의 특징 추출(ResNet)
    #RPN(Resion Proposal Networks) => Bounding Box의 후보 제안 
    #ROI Head => RPN 를 본 뒤, 분류 수행, bbox 보정**
    #Faster RCNN : 클래스 = 분류하고자 하는 객체의 개수
    #ROI Head를 변경
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #ROI Head도 2개의 부속품이 있음 -> Cls_score(분류) / BBox_predictor(바운딩박스 찾기)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 
                                                      num_classes=num_classes+1)

    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()       # 드롭아웃 / BN 학습 모드 활성화
    total_loss = 0.0

    progress_bar = tqdm(loader, desc='Train')
    for images, targets in progress_bar:
        # 이미지·타겟을 지정 디바이스로 이동
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # train 모드에서 model()은 loss_dict 반환 (box_reg, cls, rpn_cls, rpn_reg)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())   # 4개 손실 합산

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())   # tqdm 오른쪽에 현재 배치 손실 표시

    return total_loss / len(loader)   # 에포크 평균 손실

#훈련코드
def train(model, train_loader, epochs=10, lr=0.01, device=None, save_path='./faster_rcnn.pth'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.02)

    #수렴 안정화(동적 러닝스케쥴러 조정)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3)

    for e in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        print(f'[EPOCH {e+1}] avg_loss {avg_loss:.4f}')

    torch.save(model.state_dict(), save_path)
    print(f'모델 저장 완료 : {save_path}')

#검증코드
def evaluate(model, loader, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()      # BN running_mean 고정, 드롭아웃 비활성화
    model.to(device)

    results = []
    with torch.no_grad():     # 추론 시 그래디언트 계산 불필요
        for images, targets in tqdm(loader, desc='Eval'):
            images = [img.to(device) for img in images]
            # eval 모드에서 model()은 예측 dict 리스트 반환 (boxes, labels, scores)
            outputs = model(images)
            results.extend(outputs)

    return results

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Preprocessing import faster_preprocessing

if __name__ == '__main__':

    image_dir = r'C:\Users\jeong\OneDrive\Desktop\Lecture\Data\NutsDataset\images'
    label_dir = r'C:\Users\jeong\OneDrive\Desktop\Lecture\Data\NutsDataset\labels'

    train_loader, valid_loader = faster_preprocessing.get_nuts_dataloader(image_dir, label_dir)

    model = load_model(num_classes=15)

    train(model, train_loader=train_loader, epochs=1, save_path='./faster_nuts.pth')

    output = evaluate(model, valid_loader)
    print(f'훈련, 검증 완료')
