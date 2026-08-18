import torch 
import torch.nn as nn
import torch.nn.functional as F  #많이 쓰는 녀석

from tqdm import tqdm 

#기본 블록

#Down 방향 블록

#Up 방향 블록

#최종 Unet

#train
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()       #모델을 훈련 상태로 셋업
    total_loss = 0.0 

    #tqdm ->image, mask의 데이터 쌍을 입력
    for imgs, masks in tqdm(loader, desc='Train'):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)          # (B, C, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

#eval -> 평가지표가 특이!
def evaluate(model, loader, criterion, device, num_classes=16):
    model.eval()
    total_loss = 0.0
    iou_sum = torch.zeros(num_classes)
    iou_count = torch.zeros(num_classes)

    with torch.no_grad():
        for image, mask in tqdm(loader, desc='Valid'):
            image = image.to(device)
            mask = mask.to(device)

            outputs = model(image)
            loss = criterion(outputs, mask)
            total_loss += loss.item()

            #classification 예측 -> 어떤 클래스냐?
            #obejct detection 예측 -> BBox 가 어디에 있고, 그 BBOX가 무엇이냐?
            #Segmentation 예측 -> 이 덩어리(픽셀)이 무엇이냐?(픽셀의 덩어리 예측)
            preds = outputs.argmax(dim=1)  #preds 모델이 훈련 결과를 바탕으로 얻어낸 예측 값

            for cls in range(num_classes):
                pred_c = (preds == cls) #내가 예측한 픽셀
                true_c = (mask == cls)  #실제 픽셀

                #교집합 
                inter = (pred_c & true_c).sum().item() #픽셀의 겹친 부분 세기(sum.item())

                #합집합
                union = (pred_c | true_c).sum().item() #실제 픽셀이거나, 예측한 픽셀의 개수

                #겹침/전체
                if union > 0:
                    iou_sum[cls] += inter/union
                    iou_count[cls] += 1 

    mean_loss = total_loss / len(loader)

    valid_cls = iou_count > 0 #count가 0보다 큰 클래스만 체크
    mean_iou = (iou_sum[valid_cls] / iou_count[valid_cls]).mean().item()

    return mean_loss, mean_iou


