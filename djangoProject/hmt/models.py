from django.db import models


# Create your models here.
class SysModel(models.Model):
    SysModelName = models.CharField(max_length=100)
    Computation = models.FloatField(default=0.00)
    Parameter = models.FloatField(default=0.00)
    Storage = models.FloatField(default=0.00)
    Latency = models.FloatField(default=0.00)
    Energy = models.FloatField(default=0.00)
    Accuracy = models.FloatField(default=0.00)
    Infomation = models.CharField(max_length=1000)

    def __str__(self):
        return self.SysModelName

class ClassDatasetModel(models.Model):
    ClassName = models.CharField(max_length=100)
    DatasetName = models.CharField(max_length=100)
    ModelName = models.CharField(max_length=100)
    
    def __str__(self):
        return self.ClassName, self.DatasetName

class ImagesClassification(models.Model):
    
    Computation = models.FloatField(default=0.00)
    Parameter = models.FloatField(default=0.00)
    Energy = models.FloatField(default=0.00)
    Storage = models.FloatField(default=0.00)
    Accuracy = models.FloatField(default=0.00)
    CompressRate = models.FloatField(default=0.00)
    ModelName = models.CharField(max_length=100)
    DatasetName = models.CharField(max_length=100)
    
    class Meta:
        db_table = 'hmt_imagesclassification'
    
    def __str__(self):
        return self.ModelName, self.DatasetName
        
class SysDeviceLatency(models.Model):
    SysModelName = models.CharField(max_length=100)
    Device = models.CharField(max_length=100)
    Latency = models.FloatField(default=0.00)
    Energy = models.FloatField(default=0.00)

    def __str__(self):
        return self.SysModelName, self.Device
    
class Device(models.Model):
    DeviceName = models.CharField(max_length=100)
    BatteryVolume = models.IntegerField()
    CPU = models.CharField(max_length=100)
    DRam = models.IntegerField()

    def __str__(self):
        return self.DeviceName

class Mission(models.Model):
    MissionName = models.CharField(max_length=100)

    def __str__(self):
        return self.MissionName

class AbstractModel(models.Model):
    MissionName2 = models.ForeignKey('Mission', on_delete=models.PROTECT)
    Dataset = models.CharField(max_length=100)
    ModelName = models.CharField(max_length=100, null=True)
    Flops = models.FloatField(default=0.00)
    Params = models.FloatField(default=0.00)
    Storage = models.FloatField(default=0.00)
    Energy = models.FloatField(default=0.00)
    Accuracy = models.FloatField(default=0.00)
    CompressRate = models.FloatField(default=0.00)

    class Meta:
        abstract = True


class ImageClassification(AbstractModel):
    def __str__(self):
        return self.ModelName


class SemanticSegmentation(AbstractModel):
    def __str__(self):
        return self.ModelName
