import json
import shutil, os, sys
import re
import socket
import pickle
import threading
from tabnanny import check

from django.conf import settings
from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.utils.encoding import escape_uri_path
from rest_framework import status
from rest_framework import response
from rest_framework.response import Response
from rest_framework.views import APIView

from djangoProject.settings import SYSMODELDIA_ROOT, SYSMODELCODEDIA_ROOT
from djangoProject.settings import DOWNLOADFILEDIR_ROOT, UPLOADUSERMODEL_ROOT
from hmt.models import Device, Mission, ImageClassification
from hmt.serializers import DeviceSerializer, ImageClassificationSerializer

from hmt.models import SysModel, SysDeviceLatency
from hmt.serializers import SysModelSerializer, SysDeviceLatencySerializer

from operator import itemgetter
from pynvml import *



data_android = {"CPU_Arch": "armv7l", 
        "OS_Version": "Raspbian GNU/Linux 10", 
        "RAM_Total": 0, 
        "CPU_Use": "1.5", 
        "MEM_Use": 15.99888854,
        "DISK_Free": ""}

# class DeviceAndroid(APIView):
#     def post(self, request):
def android(request): 
    global data_android   
    if request.method == 'POST':
        data_android=request.body   #request.body就是获取http请求的内容,data是一个json格式的bytes对象
        # json_string = data_android.decode('utf-8')
        # data_android=json.loads(json_string)
        # keys=data_android.keys()
        # data = json.loads(data_android.decode('utf-8'))
        # print(data_android["CPU_Use"])
        print(data_android)
        return JsonResponse({"errorcode":0})# JsonResponse（）参数必须是字典对象，把其序列化为json格式，返回json格式的请求 如果参数不是Python对象，那么JsonResponse()将引发TypeError异常。
    elif request.method == 'GET':           #如果传入的参数不是一个字典对象，可以将JsonResponse()的第二个参数safe设置为False，这样JsonResponse()就可以处理其他Python对象类型，如列表、元组、数字、字符串等。但是，如果JsonResponse()的参数不是一个合法的Python对象，比如函数、类实例等，则依然会引发TypeError异常。
        print(data_android)
        return JsonResponse(json.loads(data_android))#json.load(data)就是一个json字符串反序列化为python对象
        #return JsonResponse(data)