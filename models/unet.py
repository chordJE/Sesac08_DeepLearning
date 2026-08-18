import torch 
import torch.nn as nn
import torch.nn.functional as F  #많이 쓰는 녀석

from tqdm import tqdm 

#기본 블록
class ConvBlock(nn.Module):
    #Sequential에서 실제 블록 흐름을 다 연결해놓고, forward는 명시만 진행
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                                  kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)
        

#Down 방향 블록
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

#Up 방향 블록
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 
                                     kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        #paper에서 copy and crop에 해당하는 부분
        #Skip connection
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], 
                              mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

#최종 Unet
class Unet(nn.Module):
    def __init__(self, in_channel=3, num_classes=16, base_ch=64):
        super().__init__()

        #인코더
        self.enc1 = ConvBlock(in_channel=in_channel, out_channel=base_ch)
        self.enc2 = DownBlock(base_ch, base_ch*2)
        self.enc3 = DownBlock(base_ch*2, base_ch*4)
        self.enc4 = DownBlock(base_ch*4, base_ch*8)

        #병목
        self.bottleneck = DownBlock(base_ch*8, base_ch * 16)

        #디코더
        self.dec1 = UpBlock(base_ch * 16, base_ch*8)
        self.dec2 = UpBlock(base_ch * 8, base_ch*4)
        self.dec3 = UpBlock(base_ch * 4, base_ch*2)
        self.dec4 = UpBlock(base_ch * 2, base_ch)

        #최종
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        s5 = self.dec1(x, s4)
        s6 = self.dec2(s5, s3)
        s7 = self.dec3(s6, s2)
        s8 = self.dec4(s7, s1)
        return self.head(s8)


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


#train_one_epoch, evaluate동시 실행
def train(model, train_loader, valid_loader,
          epochs=30, lr=1e-3, device=None,
          save_path='unet_nuts.pth', num_classes=16):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    best_iou = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = evaluate(model, valid_loader, criterion, device, num_classes)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"train_loss: {train_loss:.4f}  "
              f"val_loss: {val_loss:.4f}  "
              f"mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"  → 모델 저장 (best mIoU: {best_iou:.4f})")

    print(f"\n훈련 완료. 최고 mIoU: {best_iou:.4f}")