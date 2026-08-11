#라벨만들기 위한 os, json
import os, json 
import matplotlib.pyplot as plt 
import cv2 
import re


#json파일 기준으로 라벨 추출
def label_from_json(data_list):
    # i=> 한 개의 라벨
    #'Code Name': 'A220120XX_10337.jpg' -> .을 기준으로 나눠서 앞부분 가져옴 -> A220120XX_10337
    filename = data_list['Code Name']

    #너비, 높이 추출
    w = data_list['W']
    h = data_list['H']

    #x, y 센터 포인트
    x, y = data_list['Point(x,y)'].split(',')

    w = float(w)
    h = float(h)
    x = float(x)
    y = float(y)

    #print(f'{filename}에서 추출된 대상 : {x}, {y}, {w}, {h}')
    return x, y, w, h, filename

#txt파일 기준으로 라벨 추출
def label_from_txt(sample_path):
    with open(sample_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        words = []

        # ['Code', 'Name', 'A220120XX10306.jpg']
        # ['Pointxy', '0.317078189300412, 0.479356405585914']
        # ['W', '0.633333333333333'], ['H', '0.957498482088646']
        for line in lines :
            #공백제거
            parts = line.strip().split()
            words.append([re.sub(r'[^a-zA-Z0-9.,]', '', x) for x in parts])

        Width, Height = 0, 0
        point_x, point_y = 0, 0
        path=''
        for w in words:
            if 'W' in w:
                Width = w[1]

            if 'H' in w:
                Height = w[1]

            if 'Pointx,y' in w:
                point_x, point_y = w[1].split(',')[0], w[1].split(',')[1]

            if 'Code' in w:
                path = w[2]

        print(Width, Height, point_x, point_y, path)


def create_yolo_label():
    #label_folder 정해놓은 '루트 폴더' 안에 있는 모든 파일
    # 루트폴더 + 파일 1
    # 루트폴더 + 파일 2...
    # x for x in os.listdir(label_folder)
    # os.path.join(a, b) -> a와 b경로를 합쳐서 하나의 경로 변환
    # json_list = [os.path.join(label_folder, x) 
    #              for x in os.listdir(label_folder) if 'json' in x]
    # print(json_list)

    json_list = []
    for i in os.listdir(label_folder):
        path = os.path.join(label_folder, i)
        json_list.append(path)

    for i in range(len(json_list)):
        #json_list[i] -> i번째의 .json 파일
        with open(json_list[i], 'r', encoding='utf-8') as f:
            data_list = json.load(f)

            lines = []
            #data -> 한 개의 json파일 안에 있는 한 개의 라벨 
            for data in data_list:
                #하나의 라벨 덩어리에서 x, y, w, h, filename을 추출!
                x, y, w, h, filename = label_from_json(data)

                #txt 파일로 변환!
                lines.append(f'0    {x}    {y}    {w}    {h}\n')



            out_path = os.path.join(label_folder, 'yolo_txt_label')
            # '경로' 가 없을때-> os.mkdir(경로) 만들어줘! 
            if not os.path.exists(out_path):
                os.mkdir(out_path)

            #'./Data/PeachDataset/peach_label/yolo_txt_label/파일네임.txt'
            #file_path = filename+'.txt'
            #txt_path = os.path.join(out_path, file_path)
            txt_path = f'{out_path}/{filename}.txt'
            print(txt_path)

            #'파일이름'으로 lines 리스트를 txt파일로 저장
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)


if __name__ == '__main__':
    #json파일에서 '특정한 키워드'를 추출해서 가져오고 싶다.
    # image_folder = r'./Data/PeachDataset/peach_image'
    # label_folder = r'./Data/PeachDataset/peach_label'

    #같은 일련번호를 가진 복숭아 사진-라벨 가져올것!
    image_file = r'./Data/PeachDataset/peach_image/train/A220120XX_10317.jpg'
    txt_file = r'./Data/PeachDataset/peach_label/yolo_txt_label/A220120XX_10317.jpg.txt'

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]

    with open(txt_file, 'r', encoding='utf-8') as f:
        for l in f:
            label, cx, cy, w, h = l.strip().split()
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            x1 = int((cx - (w/2)) * image_w)
            y1 = int((cy - (h/2)) * image_h)
            x2 = int((cx + (w/2)) * image_w)
            y2 = int((cy + (h/2)) * image_h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    plt.imshow(image)
    plt.show()
            

#딥러닝 시퀀스
#1.데이터 가져옴
#2.데이터 정제(preprocessing)
#3.알고리즘선택
#4.훈련
#5.검증
#6.평가
#7.배포