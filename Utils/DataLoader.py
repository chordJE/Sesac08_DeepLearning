#컴퓨터(os)와 이미지 데이터 로딩
import os 
from PIL import Image
import matplotlib.pyplot as plt 

#데이터셋, 데이터 로더, 데이터로딩 시 스플릿 관련 라이브러리
from torchvision import datasets, models, transforms 
from torch.utils.data import DataLoader, Dataset, random_split 

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

#실제 훈련 가능한 상태로 데이터를 셋업하는 함수
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
