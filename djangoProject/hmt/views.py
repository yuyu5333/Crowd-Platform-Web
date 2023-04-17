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

from djangoProject.settings import MEDIA_ROOT, SYSMODELDIA_ROOT, SYSMODELCODEDIA_ROOT
from djangoProject.settings import DOWNLOADFILEDIR_ROOT, UPLOADUSERMODEL_ROOT
from hmt.models import Device, Mission, ImageClassification
from hmt.serializers import DeviceSerializer, ImageClassificationSerializer

from hmt.models import SysModel, SysDeviceLatency
from hmt.serializers import SysModelSerializer, SysDeviceLatencySerializer

from operator import itemgetter
from pynvml import *

import uuid
import torch
import time
from thop import clever_format
# from hmt.views.nodegraph import optimal
from uploadusermodel.profile_my import profile
from uploadusermodel.checkmodel_util import test
from uploadusermodel.checkmodel_util import model_user

from Luohao.optimation import readdata

# from hmt.views.nodegraph import optimal  #路径必须这么写才行,django的根目录开始，默认从django的根目录开始识别
# Create your views here.
class ReturnSysModelStatus(APIView):
    def post(self, request):
        sysmodel_obj = json.loads(request.body)
        sysmodel_name = sysmodel_obj.get('SysModelName')
        sysmodel = SysModel.objects.get(SysModelName=sysmodel_name)
        sysmodel_serializer = SysModelSerializer(sysmodel)
        return Response(sysmodel_serializer.data)


class UploadUserModel(APIView):
    def post(self, request):
        print(request.FILES)
        # 接收文件
        file_obj = request.FILES.get('file', None)
        print("file_obj", file_obj.name)

        # 文件上传至根目录/upload/json文件夹下
        head_path = UPLOADUSERMODEL_ROOT
        print("head_path", head_path)
        # 判断是否存在文件夹
        # 如果没有就创建文件路径
        if not os.path.exists(head_path):
            os.makedirs(head_path)

        # 判断文件大小不能超过5M
        if file_obj.size > 5242880:
            return JsonResponse({'status': status.HTTP_403_FORBIDDEN, 'msg': '文件过大'},
                                status=status.HTTP_403_FORBIDDEN)

        # 文件后缀
        suffix = file_obj.name.split(".").pop()
        print("文件后缀", suffix)

        # 判断文件后缀
        suffix_list = ["py"]
        if suffix not in suffix_list:
            return JsonResponse({'status': status.HTTP_403_FORBIDDEN, 'msg': '只能选择py文件'},
                                status=status.HTTP_403_FORBIDDEN)

        # # 重命名文件
        # file_name = '%s.%s' % (uuid.uuid4(), suffix)
        # print("file_name", file_name)

        # file_name = file_obj
        file_name = "UserModel.py"

        # 储存路径
        file_path = os.path.join(head_path, file_name)
        print("储存路径", file_path)

        # 写入文件到指定路径
        with open(file_path, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        data = {}
        data['name'] = file_name

        return JsonResponse({'status': status.HTTP_200_OK, 'data': data}, status=status.HTTP_200_OK)


class CheckUserModel(APIView):
    def post(self, request):

        file_name = "UserModel.py"

        getCheck = modelCheck(file_name)

        print("getCheck: ", getCheck)

        if getCheck == False:
            return({"CheckStatus": "模型检测失败"})

        print("return")

        # return JsonResponse({'status': status.HTTP_200_OK, 'data': data}, status=status.HTTP_200_OK)
        return Response({"CheckStatus": "模型检测成功"})


class ReturnUserModelStatus(APIView):
    def post(self, request):

        print("get user model status")

        # getCheck = modelCheck("UserModel.py")

        # if getCheck == True:
            # model, input = model_user()

        model, input = model_user()
        

        print("Check pass cal start")

        Macs, Params = modelCalculate(model, input)
        Latency = modelLatency(model, input)
        Storage = modelStorage(model)

        Latency = ('%.2f' % (Latency * 1000))
        Storage = ('%.2f' % Storage)

        return_data = {
            "Computation": Macs[0:-1], "Parameter": Params[0:-1], "Latency": Latency, "Storage": Storage,
            "Energy": "None", "Accuracy": "None"
        }

        print("return_data: ", return_data)

        return Response(return_data)
        

def modelStorage(model):
    torch.save(model, "./uploadusermodel_temp.pth")
    print("Saving model successfully!")
    Storage = os.path.getsize("./uploadusermodel_temp.pth")
    Storage = Storage / 2**20
    return Storage

def modelCalculate(model, input):
    Macs, Params = profile(model, inputs=(input, ))
    Macs, Params = clever_format([Macs, Params], "%.2f")
    return Macs, Params

def modelLatency(model, input):
    out = model(input)
    starttime = time.time()
    for i in range(10):
        out = model(input)
    endtime = time.time()
    Latency = (endtime - starttime) / 10
    print("Latency: ", Latency)
    return Latency

def modelCheck(filename):
    is_error = 0
    ChangeUserModelCodeName(filename)
    print("修改数据完成")
    try:
        test_result = test()
        model, input = model_user()
        x = model(input)
        print("x.size(): ", x.size())
        print("test_result: ", test_result)
    except:
        is_error = 1
        print("model test fail")
        return False
    if is_error == 0 and test_result == x.size():
        print("model test pass")
        return True
    
    print("Check pass")
    return True

def ChangeUserModelCodeName(filename):
    # 修改文件名（修改内容重新到新的文件）
    # 再去获得数据（）
    # 注意查看import的内容会不会变化（import新的模型还是旧的 需要测试）
    # 需要添加新的url和前端点击动作
    sys.path.append("../")

    newfilename = "checkmodel_util.py"

    shutil.copy("uploadusermodel/" + filename, "uploadusermodel/" + newfilename)

    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    # with open(newfilename, "r", encoding="utf-8") as f1,open("%s.bak" % newfilename, "w", encoding="utf-8") as f2:
    #     for line in f1:
    #         if old_str in line:
    #             line = line.replace(old_str, new_str)
    #         f2.write(line)
    # os.remove(file)
    # os.rename("%s.bak" % file, file)

class ReturnSysModelDeviceLatency(APIView):
    def post(self, request):
        sysmodel_obj = json.loads(request.body)
        sysmodel_name = sysmodel_obj.get('SysModelName')

        sysmodel = SysModel.objects.get(SysModelName=sysmodel_name)
        sysmodel_serializer = SysModelSerializer(sysmodel)

        sysdevecelatencys = SysDeviceLatency.objects.filter(SysModelName=sysmodel_name)
        sysdevicelatency_data = {}
        for sysdevecelatency in sysdevecelatencys:
            sysdevicelatency_serializer = SysDeviceLatencySerializer(sysdevecelatency)
            temp = sysdevicelatency_serializer.data
            temp_device = temp["Device"]
            temp_latency = temp["Latency"]
            if temp_latency is None or temp_latency == -1:
                temp_latency = "None"
            # temp_Energy = temp["Energy"]
            sysdevicelatency_data.update({temp_device : temp_latency})

        sysdevicelatency_data.update(sysmodel_serializer.data)
        

        for temp_k,temp_v in sysdevicelatency_data.items():
            if temp_v == -1 or temp_v is None:
                sysdevicelatency_data[temp_k] = "None"


        print(sysdevicelatency_data)

        return Response(sysdevicelatency_data)

class ReturnDeviceStatus(APIView):
    def post(self, request):
        device_obj = json.loads(request.body)
        device_name = device_obj.get('DeviceName')
        device = Device.objects.get(DeviceName=device_name)
        serializer = DeviceSerializer(device)
        print(serializer.data)
        return Response(serializer.data)


class ReturnMissionStatus(APIView):
    def post(self, request):
        mission_obj = json.loads(request.body)
        mission_name = mission_obj.get('MissionName')
        mission = Mission.objects.get(MissionName=mission_name[0])
        if str(mission) == 'image_classification':
            mission = ImageClassification.objects.filter(MissionName2=mission).values()
        for item in mission:
            if item['ModelName'] == mission_name[1]:
                model = ImageClassification.objects.get(ModelName=item['ModelName'])
                serializer = ImageClassificationSerializer(model)
                return Response(serializer.data)
        raise Http404


def find_closest_compress(compress_ratio, model_set):
    if compress_ratio >= model_set[-1]['CompressRate']:
        serializer = ImageClassificationSerializer(model_set[-1])
        return serializer
    elif compress_ratio <= model_set[0]['CompressRate']:
        serializer = ImageClassificationSerializer(model_set[0])
        return serializer
    pos = 0
    for i in range(len(model_set)):
        if model_set[i]['CompressRate'] >= compress_ratio:
            pos = i
            break
    before = model_set[pos - 1]['CompressRate']
    after = model_set[pos]['CompressRate']
    if after - compress_ratio < compress_ratio - before:
        serializer = ImageClassificationSerializer(model_set[pos])
    else:
        serializer = ImageClassificationSerializer(model_set[pos - 1])
    return serializer


class ReturnCompressModel(APIView):
    def post(self, request):
        compress_rate_obj = json.loads(request.body)
        compress_rate = compress_rate_obj.get('CompressRate')
        mission_name = compress_rate_obj.get('MissionName')
        mission = Mission.objects.get(MissionName=mission_name[0])
        compress_ratio = float(compress_rate)
        model_set = []
        if str(mission) == 'image_classification':
            compress_model = ImageClassification.objects.filter(MissionName2=mission).values()
        for item in compress_model:
            if str(item['ModelName']).startswith(mission_name[1]):
                model = ImageClassification.objects.get(ModelName=item['ModelName'])
                model_set.append(model.__dict__)
        model_set = sorted(model_set, key=lambda x: x['CompressRate'])
        serializer = find_closest_compress(compress_ratio, model_set)
        return Response(serializer.data)

class DownloadModeldefinition(APIView):
    def get(self, request):
        filename = request.GET.get('modeldefinition')
        filename = filename + ".md"
        print(filename)
        download_file_path = os.path.join(DOWNLOADFILEDIR_ROOT, filename)
        print("download_file_path", download_file_path)

        response = self.big_file_download(download_file_path, filename)
        if response:
            return response

        return JsonResponse({'status': 'HttpResponse', 'msg': '文件下载失败'})

    def file_iterator(self, file_path, chunk_size=512):
        """
        文件生成器,防止文件过大，导致内存溢出
        :param file_path: 文件绝对路径
        :param chunk_size: 块大小
        :return: 生成器
        """
        with open(file_path, mode='rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break

    def big_file_download(self, download_file_path, filename):
        try:
            response = StreamingHttpResponse(self.file_iterator(download_file_path))
            # 增加headers
            response['Content-Type'] = 'application/octet-stream'
            response['Access-Control-Expose-Headers'] = "Content-Disposition, Content-Type"
            response['Content-Disposition'] = "attachment; filename={}".format(escape_uri_path(filename))
            return response
        except Exception:
            return JsonResponse({'status': status.HTTP_400_BAD_REQUEST, 'msg': '文件下载失败'},
                                status=status.HTTP_400_BAD_REQUEST)



class DownloadSysModelCode(APIView):
    def get(self, request):
        filename = request.GET.get('modelcode')
        filename = filename + ".py"
        print(filename)
        download_file_path = os.path.join(SYSMODELCODEDIA_ROOT, filename)
        print("download_file_path", download_file_path)

        response = self.big_file_download(download_file_path, filename)
        if response:
            return response

        return JsonResponse({'status': 'HttpResponse', 'msg': '模型代码下载失败'})

    def file_iterator(self, file_path, chunk_size=512):
        """
        文件生成器,防止文件过大，导致内存溢出
        :param file_path: 文件绝对路径
        :param chunk_size: 块大小
        :return: 生成器
        """
        with open(file_path, mode='rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break

    def big_file_download(self, download_file_path, filename):
        try:
            response = StreamingHttpResponse(self.file_iterator(download_file_path))
            # 增加headers
            response['Content-Type'] = 'application/octet-stream'
            response['Access-Control-Expose-Headers'] = "Content-Disposition, Content-Type"
            response['Content-Disposition'] = "attachment; filename={}".format(escape_uri_path(filename))
            return response
        except Exception:
            return JsonResponse({'status': status.HTTP_400_BAD_REQUEST, 'msg': '模型代码下载失败'},
                                status=status.HTTP_400_BAD_REQUEST)



class DownloadSysModel(APIView):
    def get(self, request):
        filename = request.GET.get('model')
        filename = filename + ".pth"
        print(filename)
        download_file_path = os.path.join(SYSMODELDIA_ROOT, filename)
        print("download_file_path", download_file_path)

        response = self.big_file_download(download_file_path, filename)
        if response:
            return response

        return JsonResponse({'status': 'HttpResponse', 'msg': '模型下载失败'})

    def file_iterator(self, file_path, chunk_size=512):
        """
        文件生成器,防止文件过大，导致内存溢出
        :param file_path: 文件绝对路径
        :param chunk_size: 块大小
        :return: 生成器
        """
        with open(file_path, mode='rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break

    def big_file_download(self, download_file_path, filename):
        try:
            response = StreamingHttpResponse(self.file_iterator(download_file_path))
            # 增加headers
            response['Content-Type'] = 'application/octet-stream'
            response['Access-Control-Expose-Headers'] = "Content-Disposition, Content-Type"
            response['Content-Disposition'] = "attachment; filename={}".format(escape_uri_path(filename))
            return response
        except Exception:
            return JsonResponse({'status': status.HTTP_400_BAD_REQUEST, 'msg': '模型下载失败'},
                                status=status.HTTP_400_BAD_REQUEST)


class DownloadCompressModel(APIView):
    def get(self, request):
        filename = request.GET.get('model')
        filename = filename + ".pth"
        print(filename)
        download_file_path = os.path.join(MEDIA_ROOT, filename)
        print("download_file_path", download_file_path)

        response = self.big_file_download(download_file_path, filename)
        if response:
            return response

        return JsonResponse({'status': 'HttpResponse', 'msg': '模型下载失败'})

    def file_iterator(self, file_path, chunk_size=512):
        """
        文件生成器,防止文件过大，导致内存溢出
        :param file_path: 文件绝对路径
        :param chunk_size: 块大小
        :return: 生成器
        """
        with open(file_path, mode='rb') as f:
            while True:
                c = f.read(chunk_size)
                if c:
                    yield c
                else:
                    break

    def big_file_download(self, download_file_path, filename):
        try:
            response = StreamingHttpResponse(self.file_iterator(download_file_path))
            # 增加headers
            response['Content-Type'] = 'application/octet-stream'
            response['Access-Control-Expose-Headers'] = "Content-Disposition, Content-Type"
            response['Content-Disposition'] = "attachment; filename={}".format(escape_uri_path(filename))
            return response
        except Exception:
            return JsonResponse({'status': status.HTTP_400_BAD_REQUEST, 'msg': '模型下载失败'},
                                status=status.HTTP_400_BAD_REQUEST)

def ConnectReturnDevice(request):
    ipaddress = json.loads(request.body)
    print(ipaddress)
    ipaddress = ipaddress.get('IPaddress')
    
    model_set = []
    return Response(serializer.data)


def getCPUinfo():
    info = os.popen('lscpu')
    i = 0
    while 1:
        i = i + 1
        line = info.readline()
        if i == 1:
            CPU_Arch = line.split()[ 1 : 2 ]
        if i == 14:
            CPU_Type = line.split()[ 1 : 9 ]
            return (CPU_Arch[0],CPU_Type)

# get the type of GPU
def getGPUinfo():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    GPU_Type = nvmlDeviceGetName(handle).decode('utf-8')
    nvmlShutdown()
    return GPU_Type

# get the version of OS
def getOSversion():
    info = os.popen('head -n 1 /etc/issue')
    line = info.readline()
    return (line.split()[ 0 : 2 ])

# get physical memory
def getMemory():
    info = os.popen('free')
    i = 0
    while 1:
        i = i + 1
        line = info.readline()
        if i == 2:
            return (line.split()[ 1 : 2 ])

#get deviceinfo:CPU_Arch  CPU_Type  GPU_Type  OS_Version  RAM_Total
def get_deviceinfo(request):
    CPU_Arch,CPU_Type = getCPUinfo()
    CPU_Type = " ".join(str(i) for i in CPU_Type[1:8])

    GPU_Type = getGPUinfo()

    OS_Version = getOSversion()
    OS_Version = " ".join(str(i) for i in OS_Version[0:2])

    RAM_Total = int(getMemory()[0]) / 1024
    #data = {CPU_Arch,CPU_Type,GPU_Type,OS_Version,RAM_Total}
    return JsonResponse({
        'CPU_Arch':CPU_Arch,
        'CPU_Type':CPU_Type,
        'GPU_Type':GPU_Type,
        'OS_Version':OS_Version,
        'RAM_Total':RAM_Total
    })

#get CPU_Use
def  getCPUuse():
     return ( str (os.popen( "top -n1 | awk '/Cpu\(s\):/ {print $2}'" ).readline().strip()))

#get GPU_Use
def getGPUuse():
    # Init
    nvmlInit()
    # get the number of GPU
    deviceCount = nvmlDeviceGetCount()
    #total_memory = 0
    #total_free = 0
    total_used = 0
    gpu_name = ""
    gpu_num = deviceCount

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle).decode('utf-8')
        total_used += (info.used // 1048576) / 1024
    # shutdown
    nvmlShutdown()
    return total_used

#get Mem_Use
def getMEMuse():
    info = os.popen('free')
    i  = 0
    while  1 :
        i  = i  + 1
        line  = info.readline()
        if  i == 2 :
            return (line.split()[ 1 : 4 ])

#get DISK_free
def getDISKfree():
    info = os.popen( "df -h /" )
    i = 0
    while 1 :
        i  = i  + 1
        line  = info.readline()
        if i == 2 :
            return (line.split()[ 1 : 5 ])
#get resourceinfo:
def get_resourceinfo(request):
    CPU_Use = getCPUuse()
    GPU_Use = getGPUuse()
    MEM_Total = round(int(getMEMuse()[0])/1024,1)
    MEM_Use = round(int(getMEMuse()[1])/1024,1) / MEM_Total * 100
    DISK_Free = getDISKfree()[2].replace('G','')
    return JsonResponse({
        'CPU_Use':CPU_Use,
        'GPU_Use':GPU_Use,
        'MEM_Use':MEM_Use,
        'DISK_Free':DISK_Free,
    })
data_raspberry = {"CPU_Arch": "armv7l", 
        "OS_Version": "Raspbian GNU/Linux 10", 
        "RAM_Total": 0, 
        "CPU_Use": "1.5", 
        "MEM_Use": 15.99888854,
        "DISK_Free": ""}
# class GetRaspberry(APIView):
    # def post(self,request):
    #     data=request.body    
    #     return response.Response()

    # def get(self, request):
    #     print('GET方法')
    #     return response.Response()
    
def raspberry(request):
    global data_raspberry
    if request.method == 'POST':
        data_raspberry=request.body   #request.body就是获取http请求的内容,data是一个json格式的bytes对象
        # print(data)
        # return response.Response('我是post请求')
        return JsonResponse({"errorcode":0})# JsonResponse（）参数必须是字典对象，把其序列化为json格式，返回json格式的请求 如果参数不是Python对象，那么JsonResponse()将引发TypeError异常。
    elif request.method == 'GET':           #如果传入的参数不是一个字典对象，可以将JsonResponse()的第二个参数safe设置为False，这样JsonResponse()就可以处理其他Python对象类型，如列表、元组、数字、字符串等。但是，如果JsonResponse()的参数不是一个合法的Python对象，比如函数、类实例等，则依然会引发TypeError异常。
        print(data_raspberry)
        print(type(data_raspberry))
        return JsonResponse(json.loads(data_raspberry))#json.load(data)就是一个json字符串反序列化为python对象
        #return JsonResponse(data)

# python manage.py runserver 0.0.0.0:8000 0.0.0.0表示可以接受任何IP地址的请求（没有的话只能接受本机的请求），8000表示服务器监听的端口号，
data_jetson = {
        "DEVICE_NAME": "NVIDIA Jetson", 
        "CPU_Use": "1.5",
        "GPU_Use":'0', 
        "MEM_Use": 15.99888854,
        "DISK_Free": "75"} 

def jetson(request): 
    global data_jetson   
    if request.method == 'POST':
        data_jetson=request.body   #request.body就是获取http请求的内容,data是一个json格式的bytes对象
        print(data_jetson)
        return JsonResponse({"errorcode":0})# JsonResponse（）参数必须是字典对象，把其序列化为json格式，返回json格式的请求 如果参数不是Python对象，那么JsonResponse()将引发TypeError异常。
    elif request.method == 'GET':           #如果传入的参数不是一个字典对象，可以将JsonResponse()的第二个参数safe设置为False，这样JsonResponse()就可以处理其他Python对象类型，如列表、元组、数字、字符串等。但是，如果JsonResponse()的参数不是一个合法的Python对象，比如函数、类实例等，则依然会引发TypeError异常。
        print(data_jetson)
        return JsonResponse(json.loads(data_jetson))#json.load(data)就是一个json字符串反序列化为python对象
        #return JsonResponse(data)

data_mcu = {
        "DEVICE_NAME": "ESP-32", 
        "CPU_Use": "1.5",
        "MEM_Use": 15.99888854} 

def mcu(request): 
    global data_mcu   
    if request.method == 'POST':
        data_mcu=request.body   #request.body就是获取http请求的内容,data是一个json格式的bytes对象
        print(data_mcu)
        return JsonResponse({"errorcode":0})# JsonResponse（）参数必须是字典对象，把其序列化为json格式，返回json格式的请求 如果参数不是Python对象，那么JsonResponse()将引发TypeError异常。
    elif request.method == 'GET':           #如果传入的参数不是一个字典对象，可以将JsonResponse()的第二个参数safe设置为False，这样JsonResponse()就可以处理其他Python对象类型，如列表、元组、数字、字符串等。但是，如果JsonResponse()的参数不是一个合法的Python对象，比如函数、类实例等，则依然会引发TypeError异常。
        print(data_mcu)
        return JsonResponse(json.loads(data_mcu))#json.load(data)就是一个json字符串反序列化为python对象
        #return JsonResponse(data)

# def segmentation(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         data_device = data.get('device')
#         data_task = data.get('task')
#         data_model = data.get('model')
#         data_target = data.get('target')
#         print(data)
#         op=optimal(data_target)
#         print(op)
#         print(type(op))
#         return JsonResponse({'id':op[0],'num':op[1]})

def segmentation_latency(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # data = request.body#不知道为啥这个会报错，上边的就是对的哈哈哈哈，但是这个postman可以调通，然后前后端也能调通
        data_device = data.get('device')
        data_task = data.get('task')
        data_model = data.get('model')
        data_target = data.get('target')
        data_dataset = data.get('dataset')
        print(data)
        if data_model=="AlexNet":
            if data_dataset=="CIFAR10":
               op=readdata(26,data_target,"Luohao/files/alexnetcifar10.xlsx")
            else:
               op=readdata(42,data_target,"Luohao/files/vggcifar10.xlsx")
        else:
            if data_dataset=="CIFAR100":
               op=readdata(42,data_target,"Luohao/files/vggcifar100.xlsx")
            else:
               op=readdata(26,data_target,"Luohao/files/alexnetcifar100.xlsx")
        # if op[0]>12:
        #    op[0]=op[0]-12        
        return JsonResponse({'id':op.id,'time':op.msum,'energy':op.esum})
        # else:
        #    return JsonResponse({'id':ops[0],'num':ops[1]})