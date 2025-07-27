import imageio.v3 as iio
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


# im2 = np.array(Image.open('D:/Downloads/1.drawio.png').getdata())
im = iio.imread('D:/Downloads/im3.drawio.png')
im2 = iio.imread('D:/Downloads/im4(1).png')
newimage = np.zeros_like(im)
imnew = im[:,:,0].astype(int)
im2new = im2[:,:,0].astype(int)
# plt.imshow(im2)
# plt.show()
dif = np.abs(imnew-im2new)
for i in range(dif.shape[0]):
    for j in range(dif.shape[1]):
        # if np.sum(dif[i][j]) != 0:
        newimage[i][j] = [dif[i][j], 0, 0, 255]
        # else:
        #     newimage[i][j] = [0, 0, 0, 255]
plt.imshow(newimage)
plt.axis('off')
plt.imsave("test.png", newimage)
# fig, axs = plt.subplots(1,3)
# axs[0].imshow(im)
# axs[1].imshow(im2)
# axs[2].imshow(newimage)
plt.show()
print(im.shape)