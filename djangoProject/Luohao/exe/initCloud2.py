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
import signal
import json
import requests
from alexnet import AlexNet
from collections import OrderedDict

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="/home/nvidia/LH/exe/alexnet.pkl"
VGG16_MODEL_PATH="model/vgg16layermodel.pkl"

IP="192.168.1.112"
PORT=7897

class Data(object):

	def __init__(self, inputData, startLayer, endLayer):
		self.inputData=inputData
		self.startLayer=startLayer
		self.endLayer=endLayer

def timeout_handler(signum, frame):
    print("Server stopped")
    raise SystemExit()

def run(model, inputData, startLayer, endLayer):
	print("Cloud runs from %d to %d layers" % (startLayer, endLayer))
	outputs = model(inputData, startLayer, endLayer, False)
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

def receiveData(server, model, test_x, test_y):
	while True:
		conn,addr=server.accept()
		while True:
			lengthData=conn.recv(6)
			length=int.from_bytes(lengthData, byteorder='big')
			b=bytes()
			if length==0:
				continue
			count=0
			while True:
				value=conn.recv(length)
				b=b+value
				count+=len(value)
				if count>=length:
					break
			data=pickle.loads(b)
			# start=time.time()
			# print(start)
			outputs=run(model, data.inputData, data.startLayer, data.endLayer)
			acc=test(outputs, test_x, test_y)
			endtime=time.time()
			# print(endtime)
			print("Calculation task runs to completion,end time is:%f" % endtime)
			print("Calculation task runs to completion,test accuracy is:%f" % (acc))
			data={"endtime":endtime,"acc":acc}
			requests.get("http://192.168.31.169:8000/cog_natural/segmentationResult/",json=data)
			print("The end time has send to server" )
			# client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	        # client.connect((IP, PORT))
	        # sendData(client, outputs, data.endLayer+1, 1)

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
	torch.set_num_threads(3)
	test_x,test_y,test_l=get_data_set("test")
	test_x=torch.from_numpy(test_x[0:100]).float()
	test_y=torch.from_numpy(test_y[0:100]).long()
	print("Model loaded successfully")
	with open("/home/nvidia/LH/exe/strategy.txt",'r') as file:
		x = list(map(int, file.readline().strip().split()))
		IP0 = file.readline().strip()
		IP1 = file.readline().strip()
		IP2 = file.readline().strip()
	wait_time = 15
	signal.signal(signal.SIGALRM, timeout_handler)
	signal.alarm(wait_time)
	server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setblocking(1)
	server.bind((IP2, PORT))
	print("Cloud two startup, ready to accept tasks")
	server.listen(1)
	receiveData(server, model,test_x,test_y)




