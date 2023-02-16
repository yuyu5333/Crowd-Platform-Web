import json
import os
import re

from django.conf import settings
from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.utils.encoding import escape_uri_path
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from djangoProject.settings import MEDIA_ROOT, SYSMODELDIA_ROOT, SYSMODELCODEDIA_ROOT, DOWNLOADFILEDIR_ROOT
from hmt.models import Device, Mission, ImageClassification
from hmt.serializers import DeviceSerializer, ImageClassificationSerializer

from hmt.models import SysModel, SysDeviceLatency
from hmt.serializers import SysModelSerializer, SysDeviceLatencySerializer

from operator import itemgetter
from pynvml import *

# Create your views here.
class ReturnSysModelStatus(APIView):
    def post(self, request):
        sysmodel_obj = json.loads(request.body)
        sysmodel_name = sysmodel_obj.get('SysModelName')
        sysmodel = SysModel.objects.get(SysModelName=sysmodel_name)
        sysmodel_serializer = SysModelSerializer(sysmodel)
        return Response(sysmodel_serializer.data)

class ReturnSysDeviceLatency(APIView):
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
