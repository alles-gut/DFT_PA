import cv2
from scipy import io

def loadImage(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def showImage(namelist, imglist):
    for i in range(len(namelist)):
        name = namelist[i]
        img  = imglist[i]
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()

def loadmat(filename, feature):
    mat = io.loadmat(filename)
    return mat[feature]
