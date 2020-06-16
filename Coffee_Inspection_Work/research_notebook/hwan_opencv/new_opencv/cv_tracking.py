import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from scipy import fftpack, fft
import numpy as np
from PIL import Image
import math
import cv2
from time import sleep

def calculateVectors(src, thres = 40):
  ############################################################################
  # <이미지 노이즈 및 배경 제거>

  
  # 그레이 스케일로 변환
  gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
  # gray = src

  # 바이너리로 변환
  ret, binary = cv2.threshold(gray,thres,255, cv2.THRESH_BINARY)
  # cv2.imshow("a",binary)
  # cv2.waitKey(0)
  # binary = cv2.bitwise_not(binary)


  height = src.shape[0]
  width = src.shape[1]

  ############################################################################
  # <생두 무게 중심 구하기>
  R = list()
  for y in range(height):
      for x in range(width):
          if binary[y,x]:            
              R.append([x, y])

  # 질량의 합
  M = len(R)

  # 질량중심 벡터는 이미지를 기준으로 (x,y)
  R = np.array(R)
  R_x = R[:,0]
  R_y = R[:,1]

  R_x_sum = R_x.sum()
  R_y_sum = R_y.sum()

  center = np.round(R_x_sum/M).real , (np.round(R_y_sum/M)).real

  height_center = center[1]
  width_center = center[0]

  print('center:', height_center, width_center)


  ############################################################################
  # <객체 외부의 노이즈 데이터 처리하기>

  # 바이너리의 윤곽선 추출
  _, contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)    
  # print(contours)
  # sleep(10)

  # 생두 객체 윤곽과 윤곽벡터 구하기
  contours_leng = [len(i) for i in contours if len(i) != 252]
  # print('contours_leng: ', contours_leng)
  max = np.array(contours_leng).max()
  # print('max: ', max)
  contour_vector_from_center = list()
  contour_vector = list()
  for i in contours:
      if len(i) == max:
          # print(i)
          for point in i:
              contour_vector.append([point[0][1], point[0][0]])
              contour_vector_from_center.append([int(point[0][1]-height_center), int(point[0][0]-width_center)])
  # print('counts of contour vector: ', len(contour_vector))
  # print(contour_vector)
  # print(contour_vector_from_center)
  return contour_vector, contour_vector_from_center, src, binary

x = 6

norm = np.array(Image.open("./data/normal/"+ str(x) +".jpg"))
brok = np.array(Image.open("./data/broken/"+ str(x) +".jpg"))

_, norm1, ns, nb = np.array(calculateVectors(norm))
_, brok1, bs, bb = np.array(calculateVectors(brok))

norm1 = np.array(norm1)
len_norm1 = len(norm1)
a = np.array(range(len_norm1))
a = a.reshape(-1,1)
c = np.hstack((norm1, a))

brok1 = np.array(brok1)
len_brok1 = len(brok1)
a = np.array(range(len_brok1))
a = a.reshape(-1,1)
c = np.hstack((brok1, a))

# print(c)
# sleep(10)

# print("="*50)
# print(norm1)


cv2.imshow("ns",ns)
cv2.imshow("bs",bs)
# cv2.waitKey(0)

# len 안맞아서 끝까지 못간다
fig = plt.figure(figsize=(20,10))
for i in range(len_brok1):
  
  ax1 = plt.subplot(1,2,1)
  ax1.scatter(brok1[:, 0], brok1[:, 1])
  ax2 = plt.subplot(1,2,1)
  ax2.scatter(brok1[:i, 0], brok1[:i, 1])
  
  ax3 = plt.subplot(1,2,2)
  ax3.scatter(norm1[:, 0], norm1[:, 1])
  ax4 = plt.subplot(1,2,2)
  ax4.scatter(norm1[:i, 0], norm1[:i, 1])
  plt.show()
cv2.waitKey(0)
# brok1 = np.array(calculateVectors(brok)[1])


############################################################

# x = 13
# ##################################

# ## bean contour
# norm = cv2.imread("./data/normal/"+ str(x) +".jpg", cv2.IMREAD_COLOR)
# # brok = np.array(Image.open("./data/broken/"+ str(x) +".jpg"))

# norm1 = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY) # contour용

# ret, norm2 = cv2.threshold(norm1, 170, 255, cv2.THRESH_BINARY)

# _, contours, hierachy = cv2.findContours(norm2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print(len(contours))
# max_iter = 0
# max = 0
# for i in range(len(contours)):
#     if max < len(contours[i]):
#         max = len(contours[i])
#         max_iter = i

# norm3 = contours[i]

# print(norm3)

# cv2.waitKey(0)