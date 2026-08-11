import os #특정한 위치에 저장 
import matplotlib.pyplot as plt 

#매개변수에 =이 붙고 안붙고의 차이가 뭘까?
#매개변수 = 디폴트값 : 매개변수를 따로 지정하지 않더라도, '디폴트값'으로 세팅 해줘라!  
def draw_plot(history, save_path=r'./result.jpg'):

    #히스토리의 길이를 체크
    #히스토리 안 -> train_acc, train_loss, valid_acc, valid_loss
    epochs = range(1, len(history['train_acc'])+1)

    #왼쪽->loss, 오른쪽->acc
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, history['train_loss'], label='TR_loss')
    ax[0].plot(epochs, history['valid_loss'], label='VR_loss')
    ax[0].legend() #
    ax[0].set_title('Loss')

    ax[1].plot(epochs, history['train_acc'], label='TR_acc')
    ax[1].plot(epochs, history['valid_acc'], label='VR_acc')
    ax[1].legend() #
    ax[1].set_title('ACC(%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f'저장 완료 : {save_path}에 저장함')





