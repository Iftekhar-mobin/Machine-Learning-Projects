# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range(1,86):
    img = cv2.imread('raw/nine/nine ('+str(i)+').jpg')
    #cv2.imshow('img',img)
   # cv2.waitKey(0)
    #cv2.destroyAllWindows()
    edges = cv2.Canny(img,100,200)
    cv2.imwrite('edited/nine/nine ('+str(i)+').jpg',edges)
