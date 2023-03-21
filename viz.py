import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pdb
import os

file = '/home/pritika/layout_structure/temp/zh_train_0.jpg'

temp = cv2.imread(file)
#new
# temp = cv2.rectangle(temp, (int(41*temp.shape[1]/1000), int(31*temp.shape[0]/1000)), (int(218*temp.shape[1]/1000), int(142*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(488*temp.shape[1]/1000), int(61*temp.shape[0]/1000)), (int(897*temp.shape[1]/1000), int(148*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(41*temp.shape[1]/1000), int(194*temp.shape[0]/1000)), (int(907*temp.shape[1]/1000), int(250*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(40*temp.shape[1]/1000), int(162*temp.shape[0]/1000)), (int(189*temp.shape[1]/1000), int(175*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(51*temp.shape[1]/1000), int(255*temp.shape[0]/1000)), (int(364*temp.shape[1]/1000), int(289*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(41*temp.shape[1]/1000), int(415*temp.shape[0]/1000)), (int(939*temp.shape[1]/1000), int(753*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(350*temp.shape[1]/1000), int(776*temp.shape[0]/1000)), (int(507*temp.shape[1]/1000), int(789*temp.shape[0]/1000)),(203,192,255),15)
# temp = cv2.rectangle(temp, (int(641*temp.shape[1]/1000), int(776*temp.shape[0]/1000)), (int(759*temp.shape[1]/1000), int(791*temp.shape[0]/1000)),(203,192,255),15)

temp = cv2.rectangle(temp, (int(52*temp.shape[1]/1000), int(411*temp.shape[0]/1000)), (int(62*temp.shape[1]/1000), int(420*temp.shape[0]/1000)),(203,192,255),15)
temp = cv2.rectangle(temp, (int(125*temp.shape[1]/1000), int(410*temp.shape[0]/1000)), (int(135*temp.shape[1]/1000), int(420*temp.shape[0]/1000)),(203,192,255),15)




cv2.imwrite('temp.jpg', temp)