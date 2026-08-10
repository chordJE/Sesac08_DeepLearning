#백본(Backbone) -> 큰 딥러닝의 '척추' 역할
#RESNET -> Residual(잔차)
import torch 
import torch.nn as nn 
from torchvision import models

#ResNet 페이퍼의 Figure2에 있는 '베이직 블록'
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        #배치노말라이제이션 -> 배치 단위로 '정규화' 를 수행함
        #CIFAR10 -> 개, 고양이, 사슴, / 트럭, 오토바이, 비행기
        #배치의 입력 데이터 분포의 변화가 심해지는 경우가 있음
        #입력 데이터의 분포 변화로 학습 속도 저하되지 않도록 안정
        #특징 맵(out_channels)을 기준으로 평균/분산 정규화 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, 
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) #메모리 절약

        #숏컷 구현
        self.shortcut = nn.Sequential()
        #'잔차'는 1*1 Conv가 필요 => 레이어를 거친 F(x)와 크기가 동일
        #1*1 Conv를 활용하여 크기를 F(x)와 동일하게 세팅 
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out

    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        self.in_channels = 64
        super(ResNet, self).__init__()
        #컬러이미지를 대상으로 수행할것이라 생각
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=3,
                               stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #각 베이직 블록 레이어 쌓기
        self.layer1 = self._create_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._create_layer(block, 512, num_blocks[3], stride=2)

        #AdaptiveAvgPool -> 최종 산출되는 특징맵의 크기를 고정시켜줌    
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #Linear의 앞부분 -> 특징 * 너비 * 높이
        #16*7*7
        self.fc = nn.Linear(512*1*1, num_classes)


    def _create_layer(self, basic_block, out_channels, num_blocks, stride):
        layer = [basic_block(self.in_channels, out_channels, stride)]
        for i in range(1, num_blocks):
            layer.append(basic_block(out_channels, out_channels, stride=1))
        return layer


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = [배, 특, 너비, 높이]
        x = x.flatten(1) #flatten(start_dim=0, end_dim=-1) 
        x = self.fc(x)


#ResNet 객체 생성
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == '__main__':
    model = ResNet18()
    print(model)