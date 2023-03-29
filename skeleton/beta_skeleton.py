import nglpy
import pdb
import os
import json
import cv2


filepath = "/home/pritika/workspace/Data/FUNSD/funsd_test"
ann_dir = os.path.join(filepath, "annotations")
img_dir = os.path.join(filepath, "images")

max_neighbors = 9
beta = 1
  
for guid, file in enumerate(sorted(os.listdir(ann_dir))):
    file_path = os.path.join(ann_dir, file)
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    image_path = os.path.join(img_dir, file)
    image_path = image_path.replace("json", "png")
    img = cv2.imread(image_path)
    point_set = []
    labels = []
    for item in data["form"]:
        #identify label of set of words
        text, label, box = item["text"], item["label"], item['box']
        if text.strip() == "":
            continue
        point_set.append([int((box[0]+box[2])/2), int((box[1]+box[3])/2)])
        labels.append(label)

    aGraph = nglpy.EmptyRegionGraph(max_neighbors=max_neighbors, relaxed=True, beta=beta)
    aGraph.build(point_set)
    graph = aGraph.neighbors()

    for pt in point_set:
        img = cv2.circle(img, (pt[0],pt[1]), radius=10, color=(0, 0, 255), thickness=-1)
    # if(file=="82491256.json"):
    #     pdb.set_trace()
    for node in graph.keys():
        for con_node in graph[node]:
            img = cv2.line(img, (point_set[node][0],point_set[node][1]),(point_set[con_node][0],point_set[con_node][1]), color = (255,0,0), thickness= 1)
    cv2.imwrite(f'FUNSD_val/{file[:-4]}png', img)



