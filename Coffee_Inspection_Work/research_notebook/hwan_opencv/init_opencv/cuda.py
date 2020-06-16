
import time

import tensorflow as tf
from tensorflow.python.client import device_lib
tf.compat.v1.disable_eager_execution()
hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()

start = time.time()
# num = 1000000 # time : 0.9307534694671631
# num = 100000000 # gpu time : 7.895660877227783
num = 100000000 # cpu time : 9.267536640167236
# num = 300000000 # gpu time : 22.814529418945312
# num = 300000000 # cpu time : 22.456551551818848
abc = 0
# for i in range(num):
#     abc +=1
    

# print(sess.run(hello))
print("="*50)
print(abc)
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

# time : 7.887564420700073
# time : 7.548349618911743

device_lib.list_local_devices()
for device in device_lib.list_local_devices():
  print(device.name)
  print(device.memory_limit)