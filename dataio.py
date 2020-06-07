##dataio.py
##load and show argumented data

import cv2
import matplotlib.pyplot as plt
import os
from scipy import io

def loadImage(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def loadmat(filename, feature):
    mat = io.loadmat(filename)
    return mat[feature]

def showImage(namelist, imglist):
    fig = plt.figure()
    cols = 4
    rows = int(len(namelist) / 4)

    for i in range(cols * rows):
        imgplt = fig.add_subplot(cols, rows, i+1)
        plt.imshow(imglist[i], cmap='gray', vmin=0, vmax=1)
        imgplt.set_xlabel(namelist[i])
        imgplt.set_xticks([]), imgplt.set_yticks([])

    plt.show()

def showImageCV(namelist, imglist):
    for i in range(len(namelist)):
        name = namelist[i]
        img  = imglist[i]
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def saveImage(namelist, imglist):
    folder = 'result'
    if not os.path.exists(folder):
        os.mkdir(folder)
    for i in range(len(namelist)):
        dir = os.path.join(folder, namelist[i] + '.jpg')
        print(dir)
        img  = imglist[i]
        cv2.imwrite(dir, img)
