import io
import pymongo
import os
import gridfs 
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

conn = pymongo.MongoClient('127.0.0.1', 27017)
cnt = 0
categories = ['normal', 'broken', 'black']
for idx, category in enumerate(categories):
    path = './Final_data/' + category
    db = conn.get_database(category)
    files = os.listdir(path)
    fs = gridfs.GridFS(db, collection = category)
    
    for f in files:
        fp = open(path + '/' + f, 'rb')
        data = fp.read()
        stored = fs.put(data, filename = f)
        cnt += 1
        print(cnt)
        
