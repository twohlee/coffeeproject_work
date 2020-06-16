import io
import pymongo
import os
import gridfs 
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
 
# pymongo 객체 생성
conn = pymongo.MongoClient('127.0.0.1', 27017)

# # 데이터 저장
# categories = ['normal', 'broken', 'black']
# for idx, category in enumerate(categories):
#     path = './Rotated_data_rb/' + category
#     db = conn.get_database(category + '_rb')
#     files = os.listdir(path)
#     fs = gridfs.GridFS(db, collection = category)

#     for f in files:
#         fp = open(path + '/' + f, 'rb')
#         data = fp.read()
#         fs.put(data, filename = f)


# 데이터 로드
categories = ['normal', 'broken', 'black']
data = list()
for idx, category in enumerate(categories):
    db = conn.get_database(category + '_rb')
    fs = gridfs.GridFS(db, collection = category)
    tmp = list()
    for f in fs.find():
        image = Image.open(io.BytesIO(f.read()))
        tmp.append(image)
    data = data + tmp
print(len(data))