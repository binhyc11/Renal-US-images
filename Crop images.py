from PIL import Image
import os
key = 'D:/Dec/'
list1 = os.listdir('D:/Dec')


def cropping(path):
    im = Image.open (path, mode = 'r')
    w, h = im.size
    left1 = 0
    left2 = w/2
    top = 0
    bottom = h
    right1 = w/2
    right2 = w
    im1 = im.crop ((left1, top, right1, bottom))
    im2 = im.crop ((left2, top, right2, bottom))
    return im1, im2


for i in list1:
    path = key + str(i)
    im1, im2 = cropping(path)
    path_im1 = 'D:/Crop2/' + '1_' + i
    path_im2 = 'D:/Crop2/' + '2_' + i
    im1.save(path_im1)
    im2.save(path_im2)