#신경망 구성 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim

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