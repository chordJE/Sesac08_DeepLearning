#전이학습 모델 가중치
from torchvision.models import resnet34, ResNet34_Weights

def get_resnet_model():

    #default, v2..
    model = resnet34(weights=ResNet34_Weights.DEFAULT)

    #모델의 모든 파라미터를 '훈련 불가' 로 동결
    for p in model.parameters():
        p.requires_grad = False

    #모델의 마지막 FC 파라미터만 '훈련 가능'으로 동결 해제
    for p in model.fc.parameters():
        p.requires_grad = True    

    return model