import json
import cv2
from PIL import Image
import numpy as np
import easyocr

path = '/home/pritika/zh_val'
reader = easyocr.Reader(['ch_sim','en'], gpu =True)

with open('XFUND_table.json') as json_file:
    table_dict = json.load(json_file)

with open('XFUND_easy.json') as json_file:
    easy_dict = json.load(json_file)


def crop(file, table_box):
    im = Image.open(f'{path}/{file}')
    final_table = []
    for box in table_box:
        im_crop = im.crop((box[0],box[1],box[0]+box[2], box[1]+box[3]))
        crop_array = np.asarray(im_crop)
        results = reader.readtext(crop_array)
        if (len(results) != 0):
            final_table.append(box)
    return final_table



def IoU(box1, box2):
    
    x1, y1, w1, h1 = box1
    x3, y3, w2, h2 = box2
    x2 = x1 + w1
    y2 = y1 + h1
    x4 = x3 + w2
    y4 = y3 + h2

    if (x1 < x3 and x2 < x3) or (x3 < x1 and x4 < x1):
        return 0
    
    if (y1 < y3 and y2 < y3) or (y3 < y1 and y4 < y1):
        return 0
    
    if (x1 > x3 and x2 < x4) and (y1 > y3 and y2 < y4):
        return 0.5
    
    if (x1 < x3 and x2 > x4) and (y1 < y3 and y2 > y4):
        return 0.5

    
    x_inter1 = max(x1, x3)
    x_inter2 = min(x2, x4)
    y_inter1 = max(y1, y3)
    y_inter2 = min(y2, y4)

    width_inter = abs(x_inter1 - x_inter2)
    height_inter = abs(y_inter1 - y_inter2)
    area_inter = width_inter*height_inter

    area_1 = w1*h1
    area_2 = w2*h2

    area_union = area_1 + area_2 - area_inter

    iou = area_inter/area_union


    return iou

for file in table_dict.keys():
    table_box = crop(file, table_dict[file])
    easy_box = easy_dict[file]
    final_box = table_box

    for box in easy_box:
        flag = True
        for b in table_box:
            iou = IoU(box,b)
            if iou > 0 and iou <=1:
                flag = False
                break
        if flag:
            final_box.append(box)

    temp = cv2.imread(path+'/'+file)
    i = 0
    color = [(255,0,0),(0,255,0),(0,0,255)]
    for ele in final_box:
        temp = cv2.rectangle(temp, (ele[0],ele[1]), (ele[0]+ele[2], ele[1]+ele[3]),color[i],5)
        i += 1
        i = i % 3
    cv2.imwrite(f"combined/{file}", temp)
        
