import os
import numpy as np
import matplotlib.pyplot as plt

files = os.listdir('D:/np_ext/')
for i in files:
    img  = np.load ('D:/np_ext/' + i)
    plt.imshow (img, cmap="gray")
    plt.axis('off')
    plt.savefig('D:/ext/%s.png' %i, bbox_inches='tight',pad_inches = 0)
