from django.urls import path, re_path

from djangoProject.hmt.views.android import android
from . import  views
from .views import android
urlpatterns = [

    path('get-sysmodel/', views.ReturnSysModelDeviceLatency.as_view()),
    path('upload-usermodel/', views.UploadUserModel.as_view()),
    path('check-usermodel/', views.CheckUserModel.as_view()),
    path('get-usermodel/', views.ReturnUserModelStatus.as_view()),
    path('get-usermodelstruct/', views.ReturnUserModelStruct.as_view()),
    
    path('get-device/', views.ReturnDeviceStatus.as_view()),
    path('get-mission/', views.ReturnMissionStatus.as_view()),
    path('get-classdatasetmodel/', views.ReturnClassDatasetModel.as_view()),
    path('get-classdatasetmodelInfo/', views.ReturnClassDatasetModelInfo.as_view()),
    path('get-classdatasetcompressmodel/', views.ReturnClassDatasetCompressModel.as_view()),

    path('compress-model/', views.ReturnCompressModel.as_view()),
    
    re_path('^download-model/', views.DownloadCompressModel.as_view()),
    re_path('^download-sysmodel/', views.DownloadSysModel.as_view()),
    re_path('^download-sysmodelcode/', views.DownloadSysModelCode.as_view()),
    re_path('^download-modeldefinition/', views.DownloadModeldefinition.as_view()),
    # path('ip-connect/', views.ConnectReturnDevice.as_view())

    path('raspberry/',views.raspberry),
    path('jetson/',views.jetson),
    path('mcu/',views.mcu),
    path('android/', android),
    path('segmentationlatency/',views.segmentation_latency),

]
