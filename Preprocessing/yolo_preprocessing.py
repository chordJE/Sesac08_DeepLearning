import os, json, re
import shutil

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

def create_yolo_label(label_folder):
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

def move_datas(source_folder, destination_folder):
    file_list = []

    for i in os.listdir(source_folder):
        file_list.append(os.path.join(source_folder, i))

    for i in file_list:
        f = i.split('\\')[-1]

        print(f'{i} 에서 {os.path.join(destination_folder, f)}')
        shutil.copy2(i, os.path.join(destination_folder, f))

def create_data_directory(base_dir):
    #base_dir/images/train
    image = os.path.join(base_dir, 'images')
    label = os.path.join(base_dir, 'labels')
    image_train = os.path.join(base_dir , 'images', 'train')
    image_valid = os.path.join(base_dir , 'images', 'valid')
    label_train = os.path.join(base_dir , 'labels', 'train')
    label_valid = os.path.join(base_dir , 'labels', 'valid')

    for p in [image, label, image_train, image_valid, label_train, label_valid]:
        # p라는 경로의 폴더가 있는지 확인
        if not os.path.exists(p):
            os.mkdir(p) #mkdir (make directory) -> 해당 폴더를 생성
            print(f'{p} 경로를 생성하였음.')


def move_label_datas(source_dir, 
                     train_image_pth, valid_image_pth,
                     train_target, valid_target):
    #라벨 옮기기
    #1.어디에 있는지 원본 소스 위치 확인
    # source_dir = r'./Data/PeachDataset/peach_label/yolo_txt_label'
    source_files = [os.path.join(source_dir, x) for x in os.listdir(source_dir)]

    #2.train_목록 valid_목록 만들기
    # train_image_pth = r'./Data/PeachDataset/YoloDataset/images/train'
    # valid_image_pth = r'./Data/PeachDataset/YoloDataset/images/valid'
    train_list = [x for x in os.listdir(train_image_pth)]
    valid_list = [x for x in os.listdir(valid_image_pth)]

    count=0
    # train_target = r'./Data/PeachDataset/YoloDataset/labels/train'
    # valid_target = r'./Data/PeachDataset/YoloDataset/labels/valid'
    #3.목적지(train/valid)로 전송
    for target in source_files:
        f = target.split('\\')[-1].split('.')[0]

        for t in train_list:
            if f in t:
                #print(f'{f} -> train')
                f = f + '.txt'
                print(target, '-->', os.path.join(train_target, f))
                shutil.copy2(target, os.path.join(train_target, f))

        for v in valid_list:
            if f in v:
                f = f + '.txt'
                shutil.copy2(target, os.path.join(valid_target, f))