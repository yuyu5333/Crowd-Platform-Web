<template>
  <div>
    <div>
      <card shadow>
        <Row>
          <Col span="2">
            <p class="head-font">设备选择：</p>
          </Col>
          <Col span="6">
            <Select
              v-model="device"
              @on-change="checkDeviceChange"
              style="width: 100px; margin-right: 20px"
            >
              <Option
                v-for="item in deviceList"
                :value="item.value"
                :key="item.value"
                >{{ item.label }}</Option
              >
            </Select>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px" />
        <Row>
          <Col span="6"
            >设备名:
            <p class="text-font">{{ deviceStatus.DeviceName }}</p>
          </Col>
          <Col span="6"
            >电池容量(MAh):
            <p class="text-font">{{ deviceStatus.BatteryVolume }}</p>
          </Col>
          <Col span="6"
            >CPU:
            <p class="text-font">{{ deviceStatus.CPU }}</p>
          </Col>
          <Col span="6"
            >DRAM(GB):
            <p class="text-font">{{ deviceStatus.DRam }}</p>
          </Col>
        </Row>
      </card>
      <div>
        <br />
      </div>
      <!-- 任务选择start -->
      <card shadow>
        <Row>
          <Col span="2">
            <p class="head-font">任务选择：</p>
          </Col>
          <Col span="6">
            <Cascader
              :data="data"
              v-model="mission"
              style="width: 150px"
              @on-change="checkMissionChange"
            ></Cascader>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px" />
        <Row>
          <Col span="4"
            >计算量(MFlops):
            <p class="text-font">{{ missionStatus.Flops }}</p>
          </Col>
          <Col span="4"
            >参数量(MParams):
            <p class="text-font">{{ missionStatus.Params }}</p>
          </Col>
          <Col span="4"
            >存储量(MB):
            <p class="text-font">{{ missionStatus.Storage }}</p>
          </Col>
          <Col span="4"
            >能耗(mJ):
            <p class="text-font">{{ missionStatus.Energy }}</p>
          </Col>
          <Col span="4"
            >精度(%):
            <p class="text-font">{{ missionStatus.Accuracy }}</p>
          </Col>
        </Row>
      </card>
      <!-- 任务选择end -->
      <Divider style="margin: 12px 0px" />
      <!-- 数据集选择start -->
      <card shadow>
        <Row>
          <Col span="2">
            <p class="head-font">任务选择：</p>
          </Col>
          <Col span="6">
            <Cascader
              :data="data"
              v-model="mission"
              style="width: 150px"
              @on-change="checkMissionChange"
            ></Cascader>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px" />
        <Row>

        <div class="image-grid">
          <div class="image-class">
            <h1>图像分类</h1>
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/CIFAR10.png" alt="Image 2" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/CIFAR100.jpg" alt="Image 3" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/Cityscapes.jpg" alt="Image 4" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/COCO.png" alt="Image 5" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/CUB-200-2011.jpeg" alt="Image 6" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/EMNIST.jpg" alt="Image 7" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/FLOWERS.jpg" alt="Image 8" width="100" height="100">
          </div>
        </div>

        <div class="image-grid">
          <div class="image-class">
            <h1>图像分类</h1>
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/CIFAR10.png" alt="Image 2" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/CIFAR100.jpg" alt="Image 3" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/Cityscapes.jpg" alt="Image 4" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/COCO.png" alt="Image 5" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/CUB-200-2011.jpeg" alt="Image 6" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/EMNIST.jpg" alt="Image 7" width="100" height="100">
          </div>
          <div class="image-item">
            <img src="/images/dataset_images/FLOWERS.jpg" alt="Image 8" width="100" height="100">
          </div>
        </div>

        </Row>
      </card>
      <!-- 数据集选择end -->
    </div>
    <div>
      <br />
    </div>
    <div>
      <card size="large">
        <p class="head-font" style="color: rgb(17, 75, 218)">模型压缩</p>
        <Divider style="margin: 12px 0px" />
        <Row>
          <Col span="4" class="head-font"> 拖动以选择压缩率： </Col>
          <Col span="10">
            <Slider
              v-model="compressRate"
              style="margin-right: 40px"
              show-input
            ></Slider>
          </Col>
          <Col span="10">
            <Button @click="compressModel" type="primary">开始压缩</Button>
          </Col>
        </Row>
        <Row>
          <Row style="margin: 20px">
            <Col span="3"
              >计算量(MFlops):
              <p class="result">{{ compressStatus.Flops }}</p>
            </Col>
            <Col span="5">
              与压缩前之比：
              <p class="compare">
                {{ (compressStatus.Flops / missionStatus.Flops).toFixed(3) }}
              </p>
            </Col>
            <Col span="3"
              >参数量(MParams)：
              <p class="result">{{ compressStatus.Params }}</p>
            </Col>
            <Col span="5">
              与压缩前之比：
              <p class="compare">
                {{ (compressStatus.Params / missionStatus.Params).toFixed(3) }}
              </p>
            </Col>
            <Col span="3"
              >存储量(MB)：
              <p class="result">{{ compressStatus.Storage }}</p>
            </Col>
            <Col span="5">
              与压缩前之比：
              <p class="compare">
                {{
                  (compressStatus.Storage / missionStatus.Storage).toFixed(3)
                }}
              </p>
            </Col>
          </Row>
          <Row style="margin: 20px 20px">
            <Col span="3"
              >能耗(mJ)：
              <p class="result">{{ compressStatus.Energy }}</p>
            </Col>
            <Col span="5">
              与压缩前之比：
              <p class="compare">
                {{ (compressStatus.Energy / missionStatus.Energy).toFixed(3) }}
              </p>
            </Col>
            <Col span="3"
              >精度(%)：
              <p class="result">{{ compressStatus.Accuracy }}</p>
            </Col>
            <Col span="5">
              与压缩前之比：
              <p class="compare">
                {{
                  (compressStatus.Accuracy / missionStatus.Accuracy).toFixed(3)
                }}
              </p>
            </Col>
          </Row>
          <p class="head-font" style="margin-left: 35%; margin-bottom: 100px">
            选用的压缩算子为：{{ compressStatus.ModelName }}
          </p>
          <Button type="primary" style="margin-left: 45%">
            <a
              :href="'download-model/?model=' + compressStatus.ModelName"
              style="color: white"
              >下载模型
            </a>
          </Button>
        </Row>
      </card>
    </div>
  </div>
</template>
<script>
import axios from "axios";

export default {
  data() {
    return {
      mission: [],
      data: [
        {
          value: "image_classification",
          label: "图像分类",
          children: [
            {
              value: "Vgg16",
              label: "Vgg16",
            },
            {
              value: "AlexNet",
              label: "AlexNet",
            },
            {
              value: "ResNet18",
              label: "ResNet18",
              // disabled: true,
            },
          ],
        },
        {
          value: "semantic_segmentation",
          label: "语义分割",
          disabled: true,
          children: [],
        },
      ],
      deviceList: [
        {
          value: "Xiaomi 12",
          label: "Xiaomi 12",
        },
        {
          value: "Iphone 14",
          label: "Iphone 14",
        },
      ],
      device: [],
      compressRate: 75,
      deviceStatus: [],
      missionStatus: [],
      compressStatus: [],
    };
  },
  mounted() {},
  watch: {},
  methods: {
    checkDeviceChange(device) {
      let that = this;
      axios
        .post("get-device/", {
          DeviceName: device,
        })
        .then((response) => (that.deviceStatus = response.data));
    },
    checkMissionChange(mission) {
      let that = this;
      axios
        .post("get-mission/", {
          MissionName: mission,
        })
        .then((response) => {
          console.log(response.data);
          that.missionStatus = response.data;
        });
    },
    compressModel() {
      let that = this;
      axios
        .post("compress-model/", {
          CompressRate: this.compressRate,
          MissionName: this.mission,
        })
        .then((response) => {
          console.log(response.data);
          that.compressStatus = response.data;
        });
    },
  },
};
</script>

<style>
.head-font {
  font-weight: bolder;
  font-size: 20px;
}

.result {
  font-weight: bold;
  font-size: 17px;
  color: indianred;
}

.compare {
  font-weight: bold;
  font-size: 17px;
  color: green;
}

.text-font {
  font-weight: bold;
  font-size: 17px;
  color: #6b25d3;
}

.image-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}

.image-item {
  width: calc(8% - 5px);
  margin-top: 15px;
  margin-bottom: 15px;
}

.image-class {
  display: flex;
  align-items: center;
  margin-top: 15px;
  margin-bottom: 15px;
  margin-left: 15px;
}

</style>
