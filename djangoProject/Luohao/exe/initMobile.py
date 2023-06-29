#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
# from torchvision import transforms
import torch.utils.data as Data
from data import get_data_set
import socket
import threading
import pickle
import io
import sys
import time
import requests
import json
from alexnet import AlexNet
from collections import OrderedDict
_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="/home/linaro/LH/exe/alexnet.pkl"
VGG16_MODEL_PATH="model/vgg16layermodel.pkl"

IP="192.168.1.131"
PORT=7899
class Data(object):

	def __init__(self, inputData, startLayer, endLayer):
		self.inputData=inputData
		self.startLayer=startLayer
		self.endLayer=endLayer

def run(model, inputData, startLayer, endLayer):
	print("mobile terminal from %d to %d layer" % (startLayer, endLayer))
	outputs = model(inputData, startLayer, endLayer, False)#这个方法的执行是和alexnet定义的那个forward相关的
	return outputs

def test(outputs, test_x, test_y):
	correct_classified = 0
	total = 0
	prediction = torch.max(outputs.data, 1)
	correct_classified += np.sum(prediction[1].numpy() == test_y.numpy())
	acc=(correct_classified/len(test_x))*100
	return acc

def sendData(client, inputData, startLayer, endLayer):
	data=Data(inputData, startLayer, endLayer)
	str=pickle.dumps(data)
	client.send(len(str).to_bytes(length=6, byteorder='big'))
	client.send(str)

def connect_to_server(host, port, attempts=7, delay=1):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(attempts):
        try:
            client.connect((host, port))
            return client
        except socket.error as e:
            print(f"connection error try times: {attempt+1}/{attempts}. error information: {e}")
            time.sleep(delay)  
    return None  
# def receiveData(client, model, x, test_x, test_y):
# 	while True:
# 		lengthData=client.recv(6)
# 		length=int.from_bytes(lengthData, byteorder='big')
# 		if length==0:
# 			continue
# 		b=bytes()
# 		count=0
# 		while True:
# 			value=client.recv(length)
# 			b=b+value
# 			count+=len(value)
# 			if count>=length:
# 				break
# 		data=pickle.loads(b)
# 		if data.startLayer>=len(x):
# 			acc=test(data.inputData, test_x, test_y)
# 			end=time.time()
# 			runtime=end-start
# 			#print("Calculation task runs to completion,Response time:%f,Accuracy:%f" % (runtime, acc))
# 			print("Calculation task runs to completion,Response time:%f" % (runtime))
# 			client.close()
# 			break
# 		else:
# 			count=0
# 			for i in range(data.startLayer, len(x)):
# 				if x[i]==1:
# 					break
# 				count=i
# 			outputs=run(model, data.inputData, data.startLayer, count)
# 			if count==len(x)-1:
# 				acc=test(outputs, test_x, test_y)
# 				end=time.time()
# 				runtime=end-start
# 				#print(" Calculation task runs to completion,Response time:%f,Accuracy:%f" % (runtime, acc))
# 				print("Calculation task runs to completion,Response time:%f" % (runtime))
# 				client.close()
# 				break
# 			else:
# 				endLayer=0
# 				for i in range(count+1, len(x)):
# 					if x[i]==0:
# 						break
# 					endLayer=i
# 				sendData(client, outputs, count+1, endLayer)

if __name__=="__main__":
	model=AlexNet()
	state_dict = OrderedDict()
	new_state_dict = OrderedDict()
	state_dict = torch.load(ALEXNET_MODEL_PATH, map_location=torch.device("cpu"))    
	for k, v in state_dict.items():
		if k[0] == 'm' and k[1] == 'o':
			name = k[7:]  # remove `module.`
		else:
			name = k  # no change
		new_state_dict[name] = v    
	model.load_state_dict(new_state_dict)
	# torch.set_num_threads(3)
	test_x,test_y,test_l=get_data_set("test")
	test_x=torch.from_numpy(test_x[0:100]).float()#测试数据集只有100个数据，测试的层数是从0到12
	test_y=torch.from_numpy(test_y[0:100]).long()
	print("Model loaded successfully")
	with open("/home/linaro/LH/exe/strategy.txt",'r') as file:
		x = list(map(int, file.readline().strip().split()))
		IP0 = file.readline().strip()
		IP1 = file.readline().strip()
		IP2 = file.readline().strip()
	# server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# server.setblocking(1)
	# server.bind((IP2, 7898))
	# print("Cloud startup, ready to accept tasks")
	# server.listen(1)
	# client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# client.connect((IP1, PORT))
	client = connect_to_server(IP1, PORT)
	if client is None:
		print("can't connect to server")
	else:
	    print("The cloud connection is successful and the calculation task is ready to be submitted")
	# print("The task has been submitted to make an uninstall decision")
	#x=[1,1,1,1,1,1,1,1,1,1,1,1,1]
	# x=[0,0,0,0,0,0,0,0,0,0,0,0,1]
	#x=[1,1,0,1,1,1,0,0,0,1,1,1,1]
	print("Start running the calculation task")
	start=time.time()
	data={"start":start}
	print("Calculation task runs to start,end time is:%f" %start)
	# if x[0]==1:
	# 	count=0
	# 	for i in range(1, len(x)):
	# 		if x[i]==0:
	# 			break
	# 		count=count+1
	# start1=time.time()
	outputs=run(model, test_x, 0, x[0]-1)
	sendData(client, outputs, x[0], x[0]+x[1]-1)
	print("The data has been submitted to cloud")
	requests.post("http://192.168.31.169:8000/cog_natural/segmentationResult/",json=data)
	print("The starttime has send to server")
	# print(start1)
		# t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
		# t.start()

	# else:
	# 	count=0
	# 	for i in range(1, len(x)):
	# 		if x[i]==1:
	# 			break
	# 		count=i
	# 	outputs=run(model, test_x, 0, count)
	# 	if count==len(x)-1:
	# 		acc=test(outputs, test_x, test_y)
	# 		end=time.time()
	# 		runtime=end-start
	# 		#print("Calculation task runs to completion,Response time:%.6f,Accuracy:%f" % (runtime, acc))
	# 		print("Calculation task runs to completion,Response time:%.6f" % (runtime))
	# 		client.close()
	# 	else:
	# 		endLayer=0
	# 		for i in range(count+1, len(x)):
	# 			if x[i]==0:
	# 				break
	# 			endLayer=i
	# 		start2=time.time()
	# 		sendData(client, outputs, count+1, endLayer)
	# 		print(start2)
	# 		t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
	# 		t.start()




