import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from scipy import fftpack, fft
import numpy as np
from PIL import Image
import math
import cv2
from time import sleep

tmp = np.array([np.array([6,0]), 
                np.array([7,0]),
                np.array([8,0]),
                np.array([9,0]),
                np.array([10,0]),

                np.array([8,1]),
                np.array([9,1]),
                np.array([10,1]),

                np.array([11,2]),
                np.array([12,2]),

                np.array([12,3]),

                np.array([12,4]),
                np.array([13,4]),

                np.array([14,3]),

                np.array([14,2]),
                np.array([15,2]),

                np.array([15,1]),
                np.array([16,1]),

                np.array([16,0]),
                np.array([17,0]),

                np.array([18,-1]),

                np.array([18,-2]),

                np.array([18,-3]),
                np.array([17,-3]),

                np.array([16,-4]),
                np.array([17,-4]),

                np.array([15,-5]),
                np.array([16,-5]),

                np.array([14,-6]),

                np.array([13,-7]),
                np.array([12,-7]),
                np.array([11,-7]),

                np.array([10,-6]),

                np.array([10,-5]),
                
                np.array([10,-4]),
                np.array([9,-4]),

                np.array([9,-3]),
                np.array([8,-3]),

                np.array([7,-2]),
                np.array([8,-2]),
                np.array([9,-2]),

                np.array([8,-1]),
                np.array([7,-1]),
                
                ] )

def calculateVectors(src, thres = 40):
    ############################################################################
    # <이미지 노이즈 및 배경 제거>

    
    # 그레이 스케일로 변환
    # gray = cv2.cvtColor(src)
    gray = src

    # 바이너리로 변환
    ret, binary = cv2.threshold(gray,thres,255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

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
                contour_vector.append((point[0][1], point[0][0]))
                contour_vector_from_center.append((point[0][1]-height_center, point[0][0]-width_center))
    # print('counts of contour vector: ', len(contour_vector))
    # print(contour_vector)
    # print(contour_vector_from_center)
    return contour_vector, contour_vector_from_center, src, binary

# 기준잡기 clean ver
def find_base(ori_data):
  # ori_data = ori_data.astype(int)
  base = ori_data[ori_data[:,1]==0].max(0)
  return np.array([base[0]]), base[1]
# find_base(ori_data)

def find_close_x(ori_data, base_x_list, y, y_direction, res, cnt=0, idx = 0):
  ori_data = ori_data.tolist()
  ori_data.sort(key=lambda x: x[1])
  ori_data = np.array(ori_data)
  # print(ori_data)
  # sleep(10)

  if len(ori_data) > 0 and cnt < 3:
    print("="*50)
    print("len : " + str(len(ori_data)))
    print(ori_data)
    
    if len(base_x_list) > 1:
      for x in base_x_list:
        n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == x) )
        if len(n[0]) == 1:
          base_x = ori_data[n[0]][0][0]
          break
    
    else: # base_x_list => 1개
      n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == base_x_list[0]) )
      if len(n[0]) == 1:
        base_x = ori_data[n[0]][0][0]
      
    try:
      base = np.array([base_x, y])
    except:
      n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == base_x_list[-1]+1) )
      if len(n[0]) == 1:
        base_x = ori_data[n[0]][0][0]
        # print("우측을 보자~~")
        # print(base_x)
      else:
        n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == base_x_list[0]-1) )
        if len(n[0]) == 1:
          base_x = ori_data[n[0]][0][0]
          # print("좌측을 보자~~")
          # print(base_x)
        else:
          # return "need U-turn"
          if cnt == 0:
            y_direction *= -1
          y += y_direction
          cnt += 1

          # print("유턴 구간")
          # print(y_direction)
          # print(y)
          return find_close_x(ori_data, base_x_list, y, y_direction, cnt=cnt, idx=idx, res=res)
      base = np.array([base_x, y])
    # print("베이스 지점~~")
    # print(base)
        
    x_left = True
    x_right = True
    x_left_idx = base[0]-1
    x_right_idx = base[0]+1

    result_list = np.array([base])
    n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == base[0]) )
    ori_data = np.delete(ori_data, n[0], 0)

    while x_left:
      n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == x_left_idx) )
      if len(n[0]) == 1:
        x_left_idx -= 1
        result_list = np.append(result_list, ori_data[n[0]], axis=0)
        ori_data = np.delete(ori_data, n[0], 0)
      else:
        x_left = False
      
    while x_right:
      n = np.where( (ori_data[:,1] == y) & (ori_data[:,0] == x_right_idx) )
      if len(n[0]) == 1:
        x_right_idx += 1
        result_list = np.append(result_list, ori_data[n[0]], axis=0)
        ori_data = np.delete(ori_data, n[0], 0)
      else:
        x_right = False
    
    result_list = result_list[np.argsort(result_list[:, 0])]
    x_list = result_list[:,0]
    
    print("찾은 y 줄")
    print(y)
    print("찾은 x 리스트")
    print(x_list)
    print("찾아야할 데이터들")
    print(ori_data)
    print("="*50)
    

    for x in x_list:
        # print(type(x)) # <class 'numpy.int32'>
        # print(np.array([x, y, idx]))
        res = np.append( res, np.array([[x, y, idx]]), axis=0 )
        idx += 1

    y += y_direction

    # idx += len(x_list)


    return find_close_x(ori_data, x_list, y, y_direction, cnt=0, idx=idx, res=res)

  else: # ori_data가 비어있을 때
    if len(ori_data) == 0:
      print("총 검사 포인트 갯수")
      print(idx)
    #   return "empty ori_data!!"
      print("thank you! perfect")
      print(res)
      return res
    else:
      print(res)
      print(cnt)
      
      # return "Error: tracking stopped!!"
      return res

##################################
x = 3
##################################

## bean contour
norm = np.array(Image.open("./data/normal/"+ str(x) +".jpg"))
brok = np.array(Image.open("./data/broken/"+ str(x) +".jpg"))

norm1 = np.array(calculateVectors(norm)[1])
brok1 = np.array(calculateVectors(brok)[1])

# print(norm1)
# sleep(10)

arr = np.array([[0,0,0]]) 

print("norm1_pixels : " +  str(len(norm1)) )
norm2 = find_close_x(norm1, find_base(norm1)[0], find_base(norm1)[1], +1, res=arr, cnt=0, idx=0)
print(type(norm2))
norm2 = norm2[:,:2]
print("="*50)
print(len(brok1))
# sleep(10)


brok2 = find_close_x(brok1, find_base(brok1)[0], find_base(brok1)[1], +1, res=arr, cnt=0, idx=0)
# brok2 = brok2[:,:2]

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(2,2,1)
ax1.scatter(norm2[:, 0], norm2[:, 1])

ax2 = fig.add_subplot(2,2,2)
ax2.scatter(brok2[:, 0], brok2[:, 1])

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(norm)

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(brok)

plt.show()