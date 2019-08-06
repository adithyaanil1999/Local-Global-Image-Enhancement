import cv2
import math
import numpy as np
def psnr(img1, img2):
mse = MSE(img1, img2)
if mse == 0:
return 100
PIXEL_MAX = 255.0
return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def MSE(img1, img2):
squared_diff = (img1 - img2) ** 2
summed = np.sum(squared_diff)
num_pix = img1.shape[0] * img1.shape[1]
err = summed / num_pix
return err
img = "C:\\Users\\Adithya\\Pictures\\img\\tomato.tif"
img_inp = cv2.imread(img, 0)
copy_img= img_inp.copy()
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
r = cv2.selectROI(img_inp)
imCrop = img_inp[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
gaussian = cv2.GaussianBlur(imCrop, (9,9), 10.0)
output= cv2.addWeighted(imCrop, 1.5, gaussian, -0.5, 0, imCrop)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_inp=cv2.equalizeHist(img_inp,img_inp)
img_inp[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]=output
stack = np.hstack((copy_img, img_inp))
cv2.imshow('output', stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
d=psnr(copy_img,img_inp)
print(d)
d=MSE(copy_img,img_inp)
print(d)