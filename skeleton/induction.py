import pdb
import os
import json
import cv2


filepath = "/home/pritika/workspace/Data/FUNSD/funsd_test"
ann_dir = os.path.join(filepath, "annotations")
img_dir = os.path.join(filepath, "images")

def check_lr_tb(head_bbox, tail_bbox):
        lr = 0
        tb = 0
        if((tail_bbox[1]>=head_bbox[1] and tail_bbox[1]<=head_bbox[3]) or (tail_bbox[3]>=head_bbox[1] and tail_bbox[3]<=head_bbox[3])):
            lr = 1 #lr
        if((tail_bbox[0]>=head_bbox[0] and tail_bbox[0]<=head_bbox[2]) or (tail_bbox[2]>=head_bbox[0] and tail_bbox[2]<=head_bbox[2])):
            tb = 1 #tb
        return lr, tb
  
for guid, file in enumerate(sorted(os.listdir(ann_dir))):
    file_path = os.path.join(ann_dir, file)
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    image_path = os.path.join(img_dir, file)
    image_path = image_path.replace("json", "png")
    img = cv2.imread(image_path)
    bbox_set = []
    labels = []
    for item in data["form"]:
        #identify label of set of words
        text, label, box = item["text"], item["label"], item['box']
        if text.strip() == "":
            continue
        bbox_set.append(box)
        labels.append(label)

    for i,pt in enumerate(bbox_set):
        if(labels[i]=="other"):
            continue
        img = cv2.rectangle(img, (pt[0],pt[1]),(pt[2],pt[3]),color=(0, 0, 255), thickness=2)
    
    graph = {}

    for index1, head in enumerate(bbox_set):
        if(labels[index1]=="other"):
            continue
        graph[index1]=[]
        for index2, tail in enumerate(bbox_set):
            if(labels[index2]=="other"):
                continue
            if(index1!=index2):
                lr,tb = check_lr_tb(head,tail)
                if(lr==1 or tb==1):
                    graph[index1].append(index2)
    # for pt in bbox_set:
    #     img = cv2.circle(img, (int((pt[0]+pt[2])/2),int((pt[1]+pt[3])/2)), radius=10, color=(0, 0, 255), thickness=-1)

    for node in graph.keys():
        for con_node in graph[node]:
            img = cv2.line(img, (int((bbox_set[node][0]+bbox_set[node][2])/2),int((bbox_set[node][1]+bbox_set[node][3])/2)),(int((bbox_set[con_node][0]+bbox_set[con_node][2])/2),int((bbox_set[con_node][1]+bbox_set[con_node][3])/2)), color = (255,0,0), thickness= 1)
    cv2.imwrite(f'FUNSD_val/induction/{file[:-4]}png', img)



