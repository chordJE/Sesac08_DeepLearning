import matplotlib.pyplot as plt 
import cv2

def show_yolo_label(image_file, txt_file):
    """
    욜로 포맷의 라벨을 시각화하여 보여줌.
    image_file : 특정한 이미지 파일을 가져옴 
    txt_file   : image_file과 일련번호가 동일한 라벨 파일을 가져옴
    """
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