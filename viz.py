import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pdb
import os

file = '/home/pritika/testing_data/images/82253245_3247.png'

temp = cv2.imread(file)
#new
temp = cv2.rectangle(temp, (472 ,122),( 541, 127),(255,0,0),5)
#final
temp = cv2.rectangle(temp, (481, 154),( 504, 159),(0,0,255),5)
cv2.imwrite('temp.jpg', temp)