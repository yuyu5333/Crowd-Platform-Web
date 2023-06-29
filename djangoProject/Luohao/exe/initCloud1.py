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
from alexnet import AlexNet
from collections import OrderedDict

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="/home/pi/LH/exe/alexnet.pkl"
VGG16_MODEL_PATH="model/vgg16layermodel.pkl"

IP="192.168.1.112"
PORT=7899

class Data(object):

	def __init__(self, inputData, startLayer, endLayer):
		self.inputData=inputData
		self.startLayer=startLayer
		self.endLayer=endLayer

def timeout_handler(signum, frame):
    print("Server stopped")
    raise SystemExit()

def run(model, inputData, startLayer, endLayer):
	print("Cloud run from %d to %d layers" % (startLayer, endLayer))
	outputs = model(inputData, startLayer, endLayer, False)
	return outputs

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

def receiveData(server, model,IP):
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
			outputs=run(model, data.inputData, data.startLayer, data.endLayer)
			# client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			# client.connect((IP, 7897))
			client=connect_to_server(IP,7897)
			if client is None:
				print("can't connect to server")
			else:
				print("The cloud connection is successful and the calculation task is ready to be submitted")
			sendData(client, outputs, data.endLayer+1, 12)
			print("The data has been sent to the cloud")

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
	with open("/home/pi/LH/exe/strategy.txt",'r') as file:
		x = list(map(int, file.readline().strip().split()))
		IP0 = file.readline().strip()
		IP1 = file.readline().strip()
		IP2 = file.readline().strip()
	wait_time = 15
	signal.signal(signal.SIGALRM, timeout_handler)
	signal.alarm(wait_time)
	server=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setblocking(1)
	server.bind((IP1, PORT))
	print("Cloud one startup, ready to accept tasks")
	server.listen(1)
	receiveData(server, model, IP2)




