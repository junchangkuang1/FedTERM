# import torch
#
# # 示例Tensor
# tensor = torch.tensor([[0.1, 0.4, 0.5, 0.8, 0.9, 0.95, 1.0],
#                        [0.3, 0.2, 0.3, 0.78, 0.25, 0.45, 1.0],
#                        [0.17, 0.48, 0.54, 0.38, 0.29, 0.65, 1.0]])
#
# # 计算95百分位数
# threshold = torch.quantile(tensor, 0.05)
#
# print("95百分位数:", threshold.item())

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imshow

image = imread(r'G:\second_datasets\0004a_ct_0000_105.jpg')


fshift = np.fft.fft2(image)

magnitude_spectrum = 20 * np.log(np.abs(fshift))
# plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum')
# plt.axis('off')
# plt.show()

image_gray_fft2 = fshift.copy()
image_gray_fft2[:fshift.shape[0]//2 -1, fshift.shape[1]//2] = 1
image_gray_fft2[-fshift.shape[0]//2 -1:, fshift.shape[1]//2] = 1
# plt.figure(figsize=(7,7))
# plt.imshow(np.log(abs(image_gray_fft2)), cmap='gray')

# 使用逆傅里叶变换
inv_fshift = np.fft.ifftshift(image_gray_fft2)
filtered_gray_image = np.fft.ifft2(inv_fshift)
filtered_gray_image = np. abs(filtered_gray_image)
# 绘制原始图像和傅里叶变换后的灰度图像
fig, ax = plt.subplots( 1 , 2 , figsize=( 14 , 7 ))
ax[ 0 ].imshow(image)
ax[ 0 ].set_title( '原始图像' )
ax[ 0 ].axis( 'off' )
ax[ 1 ].imshow(filtered_gray_image, cmap= 'gray' )
ax[ 1 ].set_title( '傅里叶变换后的灰度图像' )
ax[ 1 ].axis( 'off' )
plt.show()