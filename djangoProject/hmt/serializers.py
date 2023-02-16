from dataclasses import field
from rest_framework import serializers

from hmt.models import Device, ImageClassification

from hmt.models import SysModel, SysDeviceLatency

class SysModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = SysModel
        fields = '__all__'

class SysDeviceLatencySerializer(serializers.ModelSerializer):
    class Meta:
        model = SysDeviceLatency
        fields = '__all__'

class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = '__all__'


class ImageClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageClassification
        exclude = ('MissionName2',)
