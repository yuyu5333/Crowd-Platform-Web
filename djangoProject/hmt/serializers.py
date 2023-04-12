from rest_framework import serializers

from hmt.models import Device, ImageClassification

from hmt.models import SysModel, SysDeviceLatency, ClassDatasetModel

from hmt.models import ImagesClassification

class ImagesClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImagesClassification
        # fields = ('DatasetName', 'ModelName')
        fields = '__all__'

class ClassDatasetModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassDatasetModel
        fields = '__all__'

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
