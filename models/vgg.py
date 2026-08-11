#전이학습 모델 가중치
from torchvision.models import vgg11, VGG11_Weights

def get_vgg_model():
    #vgg모델 가져오기로 결정, weights= 부분은 '전이학습' 을 세팅하는 부분
    #vgg11(weights=VGG11_Weights.DEFAULT) -> 가중치 + 모델 합치는 부분
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