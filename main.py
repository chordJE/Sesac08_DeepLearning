#모델 Zoo : https://docs.pytorch.org/serve/model_zoo.html
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt 

from torchvision import datasets, models, transforms 
from torch.utils.data import DataLoader, Dataset, random_split 

from torchvision.models import vgg11, VGG11_Weights

import os 
from PIL import Image


#데이터셋 준비!
#커스텀데이터셋 클래스 정의
#커스텀 데이터 클래스 -> init, len, getitem이 반드시 존재
class DriveDataset(Dataset):
    def __init__(self, root_dir, transforms = None):
        self.image_root = root_dir
        self.labels = []
        self.image_path = []
        self.transform = transforms

        #폴더 이름 기반의 라벨 인덱스 매핑
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls:i for i, cls in
                             enumerate(self.classes)}
        print(self.class_to_idx)

        #400장의 이미지를 저장
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(valid_extensions):
                    image_path = os.path.join(cls_dir, file_name)
                    self.image_path.append(image_path)
                    self.labels.append(self.class_to_idx[cls])
        print(f'{self.image_path[0]}의 라벨 {self.labels[0]}')

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # 이미지 불러오기 (RGB 3채널 변환)
        img_path = self.image_path[index]
        #신경망 이미지를 넣는다 -> 이미지 픽셀 값을 넣어줌
        #get_item함수가 '경로'에 있는 이미지를 읽고(cv, pillow)
        #이미지 -> 픽셀값으로 변환하고, 이 값을 return
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        #이미지를 돌려줄 때, resize, crop, gray
        if self.transform:
            image = self.transform(image)

        #개별 이미지-라벨의 쌍 / 전체 이미지 리스트[index]
        return image, label

    def hello(self):
        print('데이터셋 안녕')


def get_dataloader():
    #경로
    root_dir = r'C:\Users\jeong\OneDrive\Desktop\Lecture\Data\cifar10_samples'

    trans = transforms.Compose([
    transforms.ToTensor(), #텐서로 바꾸기,
    transforms.Normalize((0.5),(0.5)) #이미지 정규화하기,
        ])

    Data = DriveDataset(root_dir=root_dir, transforms=trans)

    train_size = int( len(Data) * 0.7 )
    valid_size = len(Data) - train_size

    train, valid = random_split(Data, [train_size, valid_size])


    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=4)
    return train_loader, valid_loader 

def get_model():
    model = vgg11(weights=VGG11_Weights.DEFAULT)

    #모델의 기억을 일부 동결, 일부 학습용으로 비워놓음
    #features 파트 가중치는 '학습 불가' 동결, 
    #classifier만 수정
    for params in model.parameters():
        #전체 가중치에 대해 동결
        params.requires_grad = False 

    #분류기(classifier)만 동결 해제
    for params in model.classifier.parameters():
        params.requires_grad = True
    return model 

#훈련 함수
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        #평균 loss, 맞춘 비율 %
        return total_loss/len(loader), correct/total * 100


#검증 함수
#@ 데코레이터 -> 함수의 역할 특정 / @classmethod, @staticmethod
@torch.no_grad()
def evaluate(model, loader, criterion, optimizer, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        #평균 loss, 맞춘 비율 %
        return total_loss/len(loader), correct/total * 100



if __name__ == '__main__':



    #get_dataloader = 데이터를 로딩하는 함수 
    train_loader, valid_loader = get_dataloader()

    #vgg11모델과 가중치를 로딩하는 함수
    model = get_model()

    #꼭 해야 할 것 -> GPU를 쓰기 위해!
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #에포크마다 train_one_epoch랑 eval 함수를 돌리면 됨!
    #오차함수, 최적화함수, 히스토리 딕셔너리 추가, 에포크 정해주기
    EPOCHS = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    history = {'train_loss':[], 'train_acc':[], 'valid_loss':[], 
                'valid_acc':[]}
    best_acc = 0.0
            
    for e in range(EPOCHS):
        #평균 loss, 맞춘 비율 %
        #model, loader, criterion, optimizer, device
        train_loss, train_acc = train_one_epoch(model, 
                                                train_loader,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                device=device)
        valid_loss, valid_acc = evaluate(model, 
                                                valid_loader,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                device=device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        if valid_acc > best_acc :
            best_acc = valid_acc
            save_path = f'./CIFAR10_VGG11_{e+1}epoch_{valid_acc:.2f}.pth' 
            torch.save(model.state_dict(), save_path)

            print(f'최고 기록 ! {e+1}회 에포크 --> {valid_acc:.2f}')

    #return history, best_acc





    





#전처리
#데이터셋로드
#훈련(def train)
#검증(def evaluate)
#시각화