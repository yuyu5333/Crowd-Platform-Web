from django.urls import path, re_path

from . import views
from hmt.myviews import android

urlpatterns = [

    path('api/get-sysmodel/', views.ReturnSysModelDeviceLatency.as_view()),
    path('api/upload-usermodel/', views.UploadUserModel.as_view()),
    path('api/check-usermodel/', views.CheckUserModel.as_view()),
    path('api/get-usermodel/', views.ReturnUserModelStatus.as_view()),
    path('api/get-usermodelstruct/', views.ReturnUserModelStruct.as_view()),

    
    path('api/get-device/', views.ReturnDeviceStatus.as_view()),
    path('api/get-mission/', views.ReturnMissionStatus.as_view()),
    path('api/get-classdatasetmodel/', views.ReturnClassDatasetModel.as_view()),
    path('api/get-classdatasetmodelInfo/', views.ReturnClassDatasetModelInfo.as_view()),
    path('api/get-classdatasetcompressmodel/', views.ReturnClassDatasetCompressModel.as_view()),
    path('api/get-cdpcompressmodel/', views.ReturnCDPCompressModel.as_view()),


    path('api/compress-model/', views.ReturnCompressModel.as_view()),
    # path('ip-connect/', views.ConnectReturnDevice.as_view())

    path('api/raspberry/',views.raspberry),
    path('api/jetson/',views.jetson),
    path('api/mcu/',views.mcu),
    path('api/android/', android.DeviceAndroid.as_view()),
    path('api/segmentationlatency/',views.segmentation_latency),

    re_path('api/^download-model/', views.DownloadCompressModel.as_view()),
    re_path('api/^download-sysmodel/', views.DownloadSysModel.as_view()),
    re_path('api/^download-sysmodelcode/', views.DownloadSysModelCode.as_view()),
    re_path('api/^download-modeldefinition/', views.DownloadModeldefinition.as_view()),
    # path('ip-connect/', views.ConnectReturnDevice.as_view())

]
