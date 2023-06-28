<template>
<div>
    <div>
        <card shadow>
            <Row>
                <Col span="3">
                <p class="head-font">已启用设备</p>
                </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
                <Col span="6">
                <Col span="4">
                <img id="dog" src="../../assets/images/raspberry.jpg" alt="dog">
                </Col>
                <Col span="20">设备名称:
                <p class="text-font">{{ OS_Version_rasp }}</p>
                </Col>
                </Col>
                <Col span="6">CPU_Use:
                <p class="text-font">{{ CPU_Use_rasp }}%</p>
                </Col>
                <Col span="6">MEM_Use:
                <p class="text-font">{{ MEM_Use_rasp }}%</p>
                </Col>
                <Col span="6">DISK_Free:
                <p class="text-font">{{ DISK_Free_rasp }}GB</p>
                </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
                <Col span="6">
                <Col span="4">
                <img id="dog" src="../../assets/images/jetson.jpg" alt="dog">
                </Col>
                <Col span="20">设备名称:
                <p class="text-font">{{ DEVICE_name_JET }}</p>
                </Col>
                </Col>
                <Col span="6">CPU_Use:
                <p class="text-font">{{ CPU_Use_JET }}%</p>
                </Col>
                <Col span="6">GPU_Use:
                <p class="text-font">{{ GPU_Use_JET }}%</p>
                </Col>
                <Col span="6">MEM_Use:
                <p class="text-font">{{ MEM_Use_JET }}%</p>
                </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
                <Col span="6">
                <Col span="4">
                <img id="dog" src="../../assets/images/android.jpg" alt="dog">
                </Col>
                <Col span="20">设备名称:
                <p class="text-font">{{ DEVICE_name_And }}</p>
                </Col>
                </Col>
                <Col span="6">CPU_Use:
                <p class="text-font">{{ CPU_Use_And }}%</p>
                </Col>
                <Col span="6">GPU_Use:
                <p class="text-font">{{ GPU_Use_And }}%</p>
                </Col>
                <Col span="6">MEM_Use:
                <p class="text-font">{{ MEM_Use_And }}GB</p>
                </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
                <Col span="6">
                <Col span="4">
                <img id="dog" src="../../assets/images/mcu.jpg" alt="dog">
                </Col>
                <Col span="20">设备名称:
                <p class="text-font">{{ DEVICE_name_mcu }}</p>
                </Col>
                </Col>
                <Col span="6">CPU_Use:
                <p class="text-font">{{ CPU_Use_mcu }}%</p>
                </Col>
                <Col span="6">MEM_Use:
                <p class="text-font">{{ MEM_Use_mcu }}%</p>
                </Col>
                <Col span="6">MEM_Free:
                <p class="text-font">{{ MEM_Free_mcu }}MB</p>
                </Col>

            </Row>
        </card>

        <div>
            <br />
        </div>

        <!-- 任务-数据集-模型选择start -->
        <card shadow>
            <Row>
                <Col span="3">
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
                <p style="margin-top: 5px" v-else>没有可用的模型</p>
                </Col>
                <Col span="2" v-if="modelsClassDataset.length > 0">
                <Button @click="getClassDatasetModelsInfo"> 查看模型性能 </Button>
                </Col>
                <!-- <Col span="4" v-if="modelsClassDataset.length > 0">
                <p style="margin-top: 5px;">
                    当前选择的模型为: {{ selectedmodelsClassDataset }}
                </p>
                </Col> -->
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

    <div><br/></div>

    <card shadow>
        <p class="head-font" style="color: rgb(17, 75, 218)">选择总体压缩率</p>
        <Divider style="margin: 12px 0px" />
        <Row>
            <Col span="3" class="head-font"> 拖动以选择压缩率： </Col>
            <Col span="12">
            <Slider v-model="compressRate" style="margin-right: 40px" show-input></Slider>
            </Col>
            <Col span="3">
            <!-- <Button @click="getCDCompressModel" type="primary">开始压缩</Button> -->

            <Button @click="getCDCompressModelandShow" type="primary">开始压缩</Button>

            </Col>
        </Row>
    </card>

    <div><br/></div>

    <card shadow>
        <p class="head-font" style="color: rgb(17, 75, 218)">自定义性能目标</p>
        <Divider style="margin: 12px 0px" />
        <!-- 选择多个目标约束给与权重 -->
        <Row>
            <Col span="3" class="head-font"> 选择多个目标约束： </Col>
            <Col span="3" class="head-font">
                <label>
                    <input type="checkbox" v-model="selectedValuePerformanceMulti" value="MinLatency" />
                    时延最小
                </label>
            </Col>
            <Col span="3" class="head-font">
                <label>
                    <input type="checkbox" v-model="selectedValuePerformanceMulti" value="MinEnergy" />
                    能耗最小
                </label>
            </Col>
            <Col span="3" class="head-font">
                <label>
                    <input type="checkbox" v-model="selectedValuePerformanceMulti" value="MinStorage" />
                    存储最小
                </label>
            </Col>
            <Col span="3" class="head-font">
                <label>
                    <input type="checkbox" v-model="selectedValuePerformanceMulti" value="MinCalculate" />
                    计算量最小
                </label>
            </Col>
            <Col span="3" class="head-font">
                <label>
                    <input type="checkbox" v-model="selectedValuePerformanceMulti" value="MaxAcc" />
                    精确度最高
                </label>
            </Col>
            <Col span="3">
            <Button @click="getCDPCompressModelandShow" type="primary">开始压缩</Button>
            </Col>
        </Row>
        <br>
        <!-- <br> -->
        <Row>
            <Col span="3" class="head-font"> 输入目标约束权重： </Col>
            <Col span="3" class="head-font">
                <input type="number" class="number-input" :disabled="!selectedValuePerformanceMulti.includes('MinLatency')" 
                    v-model="WeightLatency" @input="validateNumber" />
                    %
                <!-- <p>输入权重为: {{ WeightLatency }}</p> -->
            </Col>
            <Col span="3" class="head-font">
                <input type="number" class="number-input" :disabled="!selectedValuePerformanceMulti.includes('MinEnergy')" 
                    v-model="WeightEnergy" @input="validateNumber" />
                    %
                <!-- <p>输入权重为: {{ WeightEnergy }}</p> -->
            </Col>
            <Col span="3" class="head-font">
                <input type="number" class="number-input" :disabled="!selectedValuePerformanceMulti.includes('MinStorage')" 
                    v-model="WeightStorge" @input="validateNumber" />
                    %
                <!-- <p>输入权重为: {{ WeightStorge }}</p> -->
            </Col>
            <Col span="3" class="head-font">
                <input type="number" class="number-input" :disabled="!selectedValuePerformanceMulti.includes('MinCalculate')" 
                    v-model="WeightCal" @input="validateNumber" />
                    %
                <!-- <p>输入权重为: {{ WeightCal }}</p> -->
            </Col>
            <Col span="3" class="head-font">
                <input type="number" class="number-input" :disabled="!selectedValuePerformanceMulti.includes('MaxAcc')" 
                    v-model="WeightAcc" @input="validateNumber" />
                    %
                <!-- <p>输入权重为: {{ WeightAcc }}</p> -->
            </Col>
            <Col span="10" v-if="modelsClassDataset.length > 0">
                <p style="margin-top: 5px;">
                    <!-- 当前选择的性能为: {{ selectedValuePerformanceMulti }} -->
                    当前选择的性能为: {{ selectedValuePerformanceMulti.join(', ') }}
                </p>
            </Col>
        </Row>


    </card>

    <!-- <div><br/></div>
    <card>
        <Row>
            <compress-net ref="refCompressComponent" style="margin-left: 220px;display: flex; justify-content: center; align-items: center;"></compress-net>
        </Row>
    </card> -->


    <div><br/></div>    
    <card>
        <Row>
            <Row style="margin: 20px">
                <Col span="3">计算量(MFlops):
                <p class="result">{{ CDCompressModelStatus.Computation }}</p>
                </Col>
                <Col span="5">
                与压缩前之比：
                <p class="compare">
                    {{
                  (
                    CDCompressModelStatus.Computation /
                    modelsClassDatasetInfo.Computation
                  ).toFixed(3)
                }}
                </p>
                </Col>
                <Col span="3">参数量(MParams)：
                <p class="result">{{ CDCompressModelStatus.Parameter }}</p>
                </Col>
                <Col span="5">
                与压缩前之比：
                <p class="compare">
                    {{
                  (
                    CDCompressModelStatus.Parameter /
                    modelsClassDatasetInfo.Parameter
                  ).toFixed(3)
                }}
                </p>
                </Col>
                <Col span="3">存储量(MB)：
                <p class="result">{{ CDCompressModelStatus.Storage }}</p>
                </Col>
                <Col span="5">
                与压缩前之比：
                <p class="compare">
                    {{
                  (
                    CDCompressModelStatus.Storage /
                    modelsClassDatasetInfo.Storage
                  ).toFixed(3)
                }}
                </p>
                </Col>
            </Row>
            <Row style="margin: 10px 10px">
                <Col span="3">能耗(mJ)：
                <p class="result">{{ CDCompressModelStatus.Energy }}</p>
                </Col>
                <Col span="5">
                与压缩前之比：
                <p class="compare">
                    {{
                  (
                    CDCompressModelStatus.Energy / modelsClassDatasetInfo.Energy
                  ).toFixed(3)
                }}
                </p>
                </Col>
                <Col span="3">精度(%)：
                <p class="result">{{ CDCompressModelStatus.Accuracy }}</p>
                </Col>
                <Col span="5">
                与压缩前之比：
                <p class="compare">
                    {{
                  (
                    CDCompressModelStatus.Accuracy /
                    modelsClassDatasetInfo.Accuracy
                  ).toFixed(3)
                }}
                </p>
                </Col>
            </Row>
            <p class="head-font" style="margin-left: 35%; margin-bottom: 100px">
                选用的压缩算子为：{{ CDCompressModelStatus.ModelName }}
            </p>
        </Row>
        <Row>
            <Button type="primary" style="margin-left: 45%">
                <a :href="'download-model/?model=' + CDCompressModelStatus.ModelName" style="color: white">下载模型
                </a>
            </Button>
        </Row>
    </card>
    <div><br/></div></div>
</template>

<script scoped>
import axios from "axios";
import CompressImages from "./CompressImages.vue";
import CompressNet from "./CompressNet.vue";


export default {
    data() {
        return {

            compressRate: 75,
            compressStatus: [],

            modelsClassDataset: [],
            selectedmodelsClassDataset: "",
            selectedValueClass: "",
            selectedValueDataSet: "",
            selectedValuePerformance: "",
            selectedValuePerformanceMulti: [],

            modelsClassDatasetInfo: [],

            CDCompressModelStatus: [],

            showCompress: false,

            WeightLatency: null,
            WeightEnergy: null,
            WeightStorge: null,
            WeightCal: null,
            WeightAcc: null,

            CPU_Use_rasp: "",
            OS_Version_rasp: "",
            RAM_Total_rasp: 0.0,
            MEM_Use_rasp: 0.0,
            CPU_Arch_rasp: "",
            DISK_Free_rasp: "10",
      
            DEVICE_name_JET: "",
            CPU_Use_JET: "",
            GPU_Use_JET: "",
            MEM_Use_JET: "",
            DISK_Free_JET: "50",
            seg_id:"",
            seg_time:"",
            seg_energy:'',
      
            CPU_Use_And :"",
            DEVICE_name_And:'',
            MEM_Use_And :'',
            DISK_Free_And:'',
            GPU_Use_And: "",
      
            showModal: false,

            CPU_Use_mcu :"",
            DEVICE_name_mcu:'',
            MEM_Use_mcu :'',
            MEM_Free_mcu:'',
            CPU_Type_mcu:'',
        };
    },
    mounted() {
      this.timer = setInterval(() => {
          this.addDataRaspberry()
          this.addDataJetson()
          this.addDataAndroid()
          this.addDataMcu()
        },1000
      )
    },
    watch: {
        selectedValuePerformanceMulti(newValue) {
            if (!newValue.includes('MinLatency')) {
                this.WeightLatency = null;
            }
            if (!newValue.includes('MinEnergy')) {
                this.WeightEnergy = null;
            }
            if (!newValue.includes('MinStorage')) {
                this.WeightStorge = null;
            }
            if (!newValue.includes('MinCalculate')) {
                this.WeightCal = null;
            }
            if (!newValue.includes('MaxAcc')) {
                this.WeightAcc = null;
            }
        },
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
        validateNumber() {
            if (this.WeightLatency !== null && this.WeightLatency.toString().length > 3) {
            this.WeightLatency = parseInt(this.WeightLatency.toString().slice(0, 3));
            }
            if (this.WeightLatency !== null && this.WeightLatency < 1) {
                this.WeightLatency = 0;
            } else if (this.WeightLatency > 100) {
                this.WeightLatency = 100;
            }

            if (this.WeightEnergy !== null && this.WeightEnergy.toString().length > 3) {
            this.WeightEnergy = parseInt(this.WeightEnergy.toString().slice(0, 3));
            }
            if (this.WeightEnergy !== null && this.WeightEnergy < 1) {
                this.WeightEnergy = 0;
            } else if (this.WeightEnergy > 100) {
                this.WeightEnergy = 100;
            }

            if (this.WeightStorge !== null && this.WeightStorge.toString().length > 3) {
            this.WeightStorge = parseInt(this.WeightStorge.toString().slice(0, 3));
            }
            if (this.WeightStorge !== null && this.WeightStorge < 1) {
                this.WeightStorge = 0;
            } else if (this.WeightStorge > 100) {
                this.WeightStorge = 100;
            }

            if (this.WeightCal !== null && this.WeightCal.toString().length > 3) {
            this.WeightCal = parseInt(this.WeightCal.toString().slice(0, 3));
            }
            if (this.WeightCal !== null && this.WeightCal < 1) {
                this.WeightCal = 0;
            } else if (this.WeightCal > 100) {
                this.WeightCal = 100;
            }

            if (this.WeightAcc !== null && this.WeightAcc.toString().length > 3) {
            this.WeightAcc = parseInt(this.WeightAcc.toString().slice(0, 3));
            }
            if (this.WeightAcc !== null && this.WeightAcc < 1) {
                this.WeightAcc = 0;
            } else if (this.WeightAcc > 100) {
                this.WeightAcc = 100;
            }
        },
        triggerColorChangeAndDisappear() {
            this.$refs.refCompressComponent.changeColorAndDisappear();
        },
        getCDCompressModelandShow() {
            this.getCDCompressModel();
            //   this.showPopup();

            // 确保在 DOM 更新后调用 triggerColorChangeAndDisappear
            this.$nextTick(() => {
                this.triggerColorChangeAndDisappear();
            });
        },
        // CDP : class dataset performance
        getCDPCompressModelandShow() {
            this.getCDPCompressModel();
            //   this.showPopup();

            // 确保在 DOM 更新后调用 triggerColorChangeAndDisappear
            this.$nextTick(() => {
                this.triggerColorChangeAndDisappear();
            });
        },
        showPopup() {
            this.showCompress = true;
        },
        hidePopup() {
            this.showCompress = false;
        },
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
                    ModelName: this.selectedmodelsClassDataset,
                })
                .then((response) => {
                    this.modelsClassDatasetInfo = response.data;
                })
                .catch((error) => {
                    console.log(error);
                });
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
        getCDCompressModel() {
            if (this.selectedValueClass && this.selectedValueDataSet) {
                axios
                    .post("get-classdatasetcompressmodel/", {
                        ClassName: this.selectedValueClass,
                        DatasetName: this.selectedValueDataSet,
                        ModelName: this.selectedmodelsClassDataset,
                        CompressRate: this.compressRate,
                    })
                    .then((response) => {
                        this.CDCompressModelStatus = response.data;
                    })
                    .catch((error) => {
                        console.log(error);
                    });
            } else {
                this.CDCompressModelStatus = [];
            }
        },
        getCDPCompressModel() {
            if (this.selectedValueClass && this.selectedValueDataSet) {
                axios
                    .post("get-cdpcompressmodel/", {
                        ClassName: this.selectedValueClass,
                        DatasetName: this.selectedValueDataSet,
                        ModelName: this.selectedmodelsClassDataset,
                        CompressRate: this.compressRate,
                        // CompressPerformance: this.selectedValuePerformance,
                        CompressPerformance: this.selectedValuePerformanceMulti,
                        // weight
                        WeightLatency: this.WeightLatency,
                        WeightEnergy: this.WeightEnergy,
                        WeightStorge: this.WeightStorge,
                        WeightCal: this.WeightCal,
                        WeightAcc: this.WeightAcc,
                    })
                    .then((response) => {
                        this.CDCompressModelStatus = response.data;
                        console.log(CDCompressModelStatus);
                    })
                    .catch((error) => {
                        console.log(error);
                    });
            } else {
                this.CDCompressModelStatus = [];
            }
        },
      addDataAndroid() {
        axios
          .get("android/")
          .then((response) => {
            console.log(response.data);
            this.CPU_Use_And = parseFloat(response.data.CPU_Use).toFixed(3)
            this.DEVICE_name_And = response.data.OS_Version
            this.MEM_Use_And = parseFloat(response.data.MEM_Use).toFixed(3)
            this.DISK_Free_And = response.data.DISK_Free
            this.GPU_Use_And = parseFloat(response.data.GPU_Use)
  
          })
      },
      addDataMcu() {
        axios
          .get("mcu/")
          .then((response) => {
            console.log(response.data);
            this.CPU_Use_mcu = parseFloat(response.data.CPU_Use).toFixed(3)
            this.DEVICE_name_mcu = response.data.CPU_Arch
            this.MEM_Use_mcu = parseFloat(response.data.MEM_Use).toFixed(3)
            this.MEM_Free_mcu = parseFloat(response.data.MEM_Free).toFixed(3)
            this.CPU_Type_mcu = response.data.CPU_Type
  
          })
      },
      addDataRaspberry() {
        axios
          // 3.1url地址
          .get("raspberry/")
          // 3.2成功时回调函数
          .then((response) => {
            console.log(response);
            this.CPU_Use_rasp = parseFloat(response.data.CPU_Use).toFixed(3)
            this.OS_Version_rasp = response.data.OS_Version
            this.RAM_Total_rasp = response.data.RAM_Total
            this.MEM_Use_rasp = parseFloat(response.data.MEM_Use).toFixed(3)
            this.CPU_Arch_rasp = response.data.CPU_Arch
            this.DISK_Free_rasp = parseFloat(response.data.DISK_Free).toFixed(3)
  
          })
      },
      addDataJetson() {
        axios
          .get("jetson/")
          .then((response) => {
            console.log(response);
            this.CPU_Use_JET = parseFloat(response.data.CPU_Use).toFixed(3)
            this.DEVICE_name_JET = response.data.DEVICE_NAME
            this.MEM_Use_JET = parseFloat(response.data.MEM_Use).toFixed(3)
            this.DISK_Free_JET = parseFloat(response.data.DISK_Free).toFixed(3)
            this.GPU_Use_JET = parseFloat(response.data.GPU_Use).toFixed(3)
          })
      },
    },
    components: {
        CompressImages,
        CompressNet,
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

.number-input {
    width: 100px; 
    text-align: center;
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

.popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: white;
    padding: 20px;
    z-index: 10;
}

.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 5;
}

  #dog {
    border-radius: 50px;
    height: 50px;
    width: 50px;
    filter: grayscale(0%);
  }

</style>
