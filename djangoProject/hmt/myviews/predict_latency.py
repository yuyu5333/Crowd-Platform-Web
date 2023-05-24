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


data_clock_divice = {
    "Jetson Nx": 1.1 * (10 ** 9)
}


class CompressUserInfo(APIView):
    
    global data_clock_divice
    
    def get(self, request):
        
        userlatency = json.loads(request.body)
        sysmodel_name = sysmodel_obj.get('CompressUserInfo')
        
        
        
        return JsonResponse(None)