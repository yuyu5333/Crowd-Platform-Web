<template>
<div>
    <div>
        <card shadow>
            <Row>
                <Col span="2">
                <p class="head-font">设备选择：</p>
                </Col>
                <Col span="6">
                <Select v-model="device" @on-change="checkDeviceChange" style="width: 100px; margin-right: 20px">
                    <Option v-for="item in deviceList" :value="item.value" :key="item.value">{{ item.label }}</Option>
                </Select>
                </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
                <Col span="6">设备名:
                <p class="text-font">{{ deviceStatus.DeviceName }}</p>
                </Col>
                <Col span="6">电池容量(MAh):
                <p class="text-font">{{ deviceStatus.BatteryVolume }}</p>
                </Col>
                <Col span="6">CPU:
                <p class="text-font">{{ deviceStatus.CPU }}</p>
                </Col>
                <Col span="6">DRAM(GB):
                <p class="text-font">{{ deviceStatus.DRam }}</p>
                </Col>
            </Row>
        </card>
        <div>
            <br />
        </div>

        <!-- 任务-数据集-模型选择start -->
        <card shadow>
            <Row>
                <Col span="2">
                <p class="head-font">任务场景选择：</p>
                </Col>
                <Col span="2">
                <p class="head-font" style="color: fuchsia">
                    {{ selectedValueClass }}
                </p>
                </Col>
                <Col span="2">
                <p class="head-font">数据集选择：</p>
                </Col>
                <Col span="2">
                <p class="head-font" style="color: fuchsia">
                    {{ selectedValueDataSet }}
                </p>
                </Col>
                <Col span="2">
                <p class="head-font">模型选择：</p>
                </Col>
                <Col span="2">
                <Select v-if="modelsClassDataset.length > 0" style="width: 100px" v-model="selectedmodelsClassDataset">
                    <Option v-for="(item_modelname, index) in modelsClassDataset" :key="index" :value="item_modelname">{{ item_modelname }}</Option>
                </Select>
                <p style="margin-top: 5px;" v-else>没有可用的模型</p>
                </Col>
                <Col span="2" v-if="modelsClassDataset.length > 0">
                <Button @click="getClassDatasetModelsInfo"> 查看模型性能 </Button>
                </Col>
                <Col span="4" v-if="modelsClassDataset.length > 0">
                <p style="margin-top: 5px;">
                    当前选择的值为: {{ selectedmodelsClassDataset }}
                </p>
                </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <!-- 模型信息start -->
            <!-- <card shadow v-if="modelsClassDataset.length > 0"> -->
            <Row>
                <Col span="4">计算量(MFlops):
                    <p class="text-font">{{ modelsClassDatasetInfo.Computation }}</p>
                </Col>
                <Col span="4">参数量(MParams):
                    <p class="text-font">{{ modelsClassDatasetInfo.Parameter }}</p>
                </Col>
                <Col span="4">存储量(MB):
                    <p class="text-font">{{ modelsClassDatasetInfo.Storage }}</p>
                </Col>
                <Col span="4">能耗(mJ):
                    <p class="text-font">{{ modelsClassDatasetInfo.Energy }}</p>
                </Col>
                <Col span="4">精度(%):
                    <p class="text-font">{{ modelsClassDatasetInfo.Accuracy }}</p>
                </Col>
            </Row>
        </card>
        <!-- 模型信息end -->
        <div>
            <br />
        </div>
        <card shadow>
            <Row>
                <!-- 数据集以及分类 -->
                <compress-images v-bind:selected-value-class="selectedValueClass" v-bind:selected-value-dataset="selectedValueDataSet" v-on:update-selected-value-class="selectedValueClass = $event" v-on:update-selected-value-dataset="selectedValueDataSet = $event">
                </compress-images>
            </Row>
        </card>
        <!-- 数据集选择end -->
    </div>
    <div>
        <card size="large">
            <p class="head-font" style="color: rgb(17, 75, 218)">模型压缩</p>
            <Divider style="margin: 12px 0px" />
            <Row>
                <Col span="4" class="head-font"> 拖动以选择压缩率： </Col>
                <Col span="10">
                <Slider v-model="compressRate" style="margin-right: 40px" show-input></Slider>
                </Col>
                <Col span="10">
                <Button @click="compressModel" type="primary">开始压缩</Button>
                </Col>
            </Row>
            <Row>
                <Row style="margin: 20px">
                    <Col span="3">计算量(MFlops):
                    <p class="result">{{ compressStatus.Flops }}</p>
                    </Col>
                    <Col span="5">
                    与压缩前之比：
                    <p class="compare">
                        {{ (compressStatus.Flops / missionStatus.Flops).toFixed(3) }}
                    </p>
                    </Col>
                    <Col span="3">参数量(MParams)：
                    <p class="result">{{ compressStatus.Params }}</p>
                    </Col>
                    <Col span="5">
                    与压缩前之比：
                    <p class="compare">
                        {{ (compressStatus.Params / missionStatus.Params).toFixed(3) }}
                    </p>
                    </Col>
                    <Col span="3">存储量(MB)：
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
                    <Col span="3">能耗(mJ)：
                    <p class="result">{{ compressStatus.Energy }}</p>
                    </Col>
                    <Col span="5">
                    与压缩前之比：
                    <p class="compare">
                        {{ (compressStatus.Energy / missionStatus.Energy).toFixed(3) }}
                    </p>
                    </Col>
                    <Col span="3">精度(%)：
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
                    <a :href="'download-model/?model=' + compressStatus.ModelName" style="color: white">下载模型
                    </a>
                </Button>
            </Row>
        </card>
    </div>
</div>
</template>

<script>
import axios from "axios";
import CompressImages from "./CompressImages.vue";

export default {
    data() {
        return {
            data: [{
                    value: "image_classification",
                    label: "图像分类",
                    children: [{
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
            deviceList: [{
                    value: "Xiaomi 12",
                    label: "Xiaomi 12",
                },
                {
                    value: "Iphone 14",
                    label: "Iphone 14",
                },
            ],
            device: [],
            deviceStatus: [],

            mission: [],
            missionStatus: [],

            compressRate: 75,
            compressStatus: [],

            modelsClassDataset: [],
            selectedmodelsClassDataset: "",
            selectedValueClass: "",
            selectedValueDataSet: "",

            modelsClassDatasetInfo: [],
        };
    },
    mounted() {},
    watch: {
        modelsClassDataset() {
            // 如果 originalModelsClassDataset 的值发生了变化，说明 modelsClassDataset 的值也发生了变化
            // 在这里更新 selectedModel 的值
            this.selectedmodelsClassDataset = "";
        },
        selectedValueClass() {
            this.getClassDatasetModelsName();
            this.freshSelsectClassDatasetModelsName();
        },
        selectedValueDataSet() {
            this.getClassDatasetModelsName();
        },
    },
    methods: {
        freshSelsectClassDatasetModelsName() {
            // 根据 selectedValueClass 和 selectedValueDataSet，获取可选的模型列表
            // 然后更新 models 变量
            this.modelsClassDataset = [];
        },
        getClassDatasetModelsInfo() {
            axios
                .post("get-classdatasetmodelInfo/", {
                    ClassName: this.selectedValueClass,
                    DatasetName: this.selectedValueDataSet,
                    ModelName: this.selectedmodelsClassDataset
                })
                .then((response) => {
                    this.modelsClassDatasetInfo = response.data;
                })
                .catch((error) => {
                    console.log(error);
                })
        },
        getClassDatasetModelsName() {
            if (this.selectedValueClass && this.selectedValueDataSet) {
                axios
                    .post("get-classdatasetmodel/", {
                        ClassName: this.selectedValueClass,
                        DatasetName: this.selectedValueDataSet,
                    })
                    .then((response) => {
                        this.modelsClassDataset = response.data;
                    })
                    .catch((error) => {
                        console.log(error);
                    });
            } else {
                this.modelsClassDataset = [];
            }
        },
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
    components: {
        CompressImages,
    },
};
</script>

<style scoped>
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
    text-align: center;
}

.image-item {
    width: calc(10%);
    margin-top: 15px;
    margin-bottom: 15px;
    align-items: center;
    justify-content: center;
}

.image-class {
    display: flex;
    align-items: center;
    margin-top: 15px;
    margin-bottom: 30px;
    margin-left: 15px;

    font-size: 23px;
}

.image-info {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 15px;
    font-size: 17px;
}

img {
    vertical-align: middle;
    border-radius: 50%;
    width: 110px;
    height: 110px;
}
</style>
