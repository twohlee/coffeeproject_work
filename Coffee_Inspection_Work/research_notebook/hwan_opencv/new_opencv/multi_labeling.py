import cv2

"""
추가 사항 :
1. 겉에 테두리 씌우기
2. 64*64 reshape (목표 크기로)
"""

# src_path = "./img/bean/multi03.png"
src_path = "./img/bean/broken.png"

ori = cv2.imread(src_path, cv2.IMREAD_COLOR)
src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("ori", ori)

_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
src_bin = cv2.bitwise_not(src_bin)
cv2.imshow("src_bin_not",src_bin)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for i in range(1,nlabels):
    stat = stats[i]

    x = stat[0]
    y = stat[1]
    width = stat[2]
    height = stat[3]
    n_pixel = stat[4]

    if n_pixel < 20: continue
    # 최대 픽셀 지정해서 임계값보다 크면 패스해버리기

    cv2.rectangle(dst, (x, y), (x+width, y+height), (0,255,255))
    
cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
