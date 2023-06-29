<template>
    <div>
      <div>
        <card shadow>
          <Row>
            <Col span="3">
            <p class="head-font">已启用设备</p>
            <!-- <Button  type="primary" @click="addData_jetson">感知模型</Button> -->
            </Col>
            <!-- <Col span="6">
              <Select v-model="device" @on-change="checkDeviceChange" style="width:100px;margin-right: 20px">
                <Option v-for="item in deviceList" :value="item.value" :key="item.value">{{ item.label }}</Option>
              </Select>
            </Col> -->
          </Row>
          <Divider style="margin: 12px 0px" />
          <Row>
            <Col span="6">
            <Col span="6">
            <img id="dog" src="../../assets/images/rasp3.jpg" alt="dog">
            </Col>
            <Col span="18">设备名称:
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
            <Col span="6">
            <img id="dog" src="../../assets/images/jetson2.jpg" alt="dog">
            </Col>
            <Col span="18">设备名称:
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
            <Col span="6">
            <img id="dog" src="../../assets/images/jiqi1.jpg" alt="dog">
            </Col>
            <Col span="18">设备名称:
            <p class="text-font"> Drone</p>
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
            <Col span="6">
            <img id="dog" src="../../assets/images/MCU5.png" alt="dog">
            </Col>
            <Col span="18">设备名称:
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
          <br>
        </div>
      </div>
      <div>
        <br>
      </div>
      <card shadow>
        <Row>
          <Col span="3">
          <p class="head-font">模型选择：</p>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px" />
        <Form model="formItem" id="head">
          <FormItem label="设备选择">
            <CheckboxGroup v-model="formItem.device" class="head-font" size="large">
              <Checkbox label="raspberry">设备1</Checkbox>
              <Checkbox label="jetson">设备2</Checkbox>
              <Checkbox label="android">设备3</Checkbox>
            </CheckboxGroup>
          </FormItem>
          <FormItem label="系统模型" class="head-font">
            <RadioGroup v-model="formItem.model" size="large">
              <Radio label="AlexNet">AlexNet</Radio>
              <Radio label="VGG">VGG</Radio>
              <Radio label="ResNet">ResNet</Radio>
            </RadioGroup>
          </FormItem>
          <FormItem label="任务分类" class="head-font">
            <RadioGroup v-model="formItem.task" size="large">
              <Radio label="Image_classification">图像分类</Radio>
              <Radio label="Object_detection">目标检测</Radio>
            </RadioGroup>
          </FormItem>
          <FormItem label="目标约束" class="head-font">
            <RadioGroup v-model="formItem.target" size="large">
              <Radio label="latency">时延最小</Radio>
              <Radio label="energy">能耗最小</Radio>
            </RadioGroup>
          </FormItem>
          <FormItem label="数据集" class="head-font">
            <RadioGroup v-model="formItem.dataset" size="large">
              <Radio label="CIFAR10">CIFAR-10</Radio>
              <Radio label="CIFAR100">CIFAR-100</Radio>
              <Radio label="COCO">COCO</Radio>
            </RadioGroup>
          </FormItem>
          <br>
          <Button type="primary" @click="segmentation">推理评估</Button>
          <!-- <Button type="primary" style="margin-left: 33px">模型推理</Button> -->
        </Form>
      </card>
      <div>
        <br>
      </div>
      <div v-if="this.showflag">
        <div style="display:inline-block;"><img src="../../assets/images/raspberry.jpg" style="width:300px;height:300px"
            alt="1"></div>
        <div style="display:inline-block;width:750px" id="arrowImg"><img src="../../assets/images/arrow.png"
            :style="{ marginBottom: 100 + 'px', marginLeft: this.leftMargin + 'px' ,width:100+'px'}" @load="customFunc" alt="1"></div>
        <div style="display:inline-block;"><img src="../../assets/images/jetson.jpg" style="width:300px;height:300px"
            alt="1">
        </div>
      </div>
      <card shadow>
        <Row>
          <Col span="3">
          <p class="head-font">推理结果：</p>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px" />
        <Row>
          <Col span="6">模型名称:
          <p class="text-font">{{ seg_model }}AlexNet</p>
          </Col>
          <Col span="6">分割策略:
          <p class="text-font">在{{ seg_device }}第{{ seg_id }}层分割</p>
          </Col>
          <Col span="6">推理能耗:
          <p class="text-font">{{ seg_energy }} J</p>
          </Col>
          <Col span="6">推理时间:
          <p class="text-font">{{ seg_time }} 秒</p>
          </Col>
        </Row>
        <!-- <div style="display: flex; justify-content: center; align-items: center; margin-left: -200px;">
          <cartoon3 />
        </div> -->
        <div v-if="showModal" class="modal">
          <div class="modal-content">
            <cartoon3 />
            <button @click="closeModal">Close</button>
          </div>
        </div>
      </card>
    </div>

  </template>
  <script scoped>
  import axios from "axios";
  import cartoon3 from "./cartoon3.vue";
  // import modal from "./modal.vue";
  export default {
    data() {
      return {
        showflag: false,
        device: [],
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
        formItem: {
          device: [],
          task: [],
          model: [],
          target: [],
          dataset:[]
        }
      }
    },
    mounted() {
      this.timer = setInterval(() => {
          this.addData()
          this.addData_jetson()
          this.android()
          this.mcu()
        },1000
      )
    },
    watch: {},
    components: {
    cartoon3
    // modal,
  },
    methods: {
      // 3.发送axios无参数请求
      addData() {
        axios
          // 3.1url地址
          .get("raspberry/")
          // 3.2成功时回调函数
          .then((response) => {
            console.log(response);
            // this.CPU_Use.push(parseFloat(response.data.CPU_Use));
            // this.DISK_Free.push(parseFloat(response.data.DISK_Free));
            // this.GPU_Use.push(parseFloat(response.data.GPU_Use).toFixed(3));
            this.CPU_Use_rasp = response.data.CPU_Use
            this.OS_Version_rasp = response.data.OS_Version
            this.RAM_Total_rasp = response.data.RAM_Total
            this.MEM_Use_rasp = parseFloat(response.data.MEM_Use).toFixed(3)
            this.CPU_Arch_rasp = response.data.CPU_Arch
            this.DISK_Free_rasp = response.data.DISK_Free
  
            // this.DISK_Free
            // this.GPU_Use.p
          })
        // //3.3失败时回调函数
        // .catch((err) => {
        //   console.log(err);
        // });
      },
      addData_jetson() {
        axios
          .get("jetson/")
          .then((response) => {
            console.log(response);
            this.CPU_Use_JET = response.data.CPU_Use
            this.DEVICE_name_JET = response.data.DEVICE_NAME
            this.MEM_Use_JET = parseFloat(response.data.MEM_Use).toFixed(3)
            this.DISK_Free_JET = response.data.DISK_Free
            this.GPU_Use_JET = response.data.GPU_Use
          })
      },
      segmentation() {
        this.showModal = true;
        setTimeout(() => {
                this.showModal = false;
                }, 6000);
        axios
          .post("segmentationlatency/", {
            device: this.formItem.device,
            task: this.formItem.task,
            target: this.formItem.target,
            model: this.formItem.model,
            dataset:this.formItem.dataset
          })
          .then((response) => {
            console.log(response.data);
            this.seg_id = response.data.id-12
            this.seg_time = response.data.time
            this.seg_energy = response.data.energy
            // this.seg_device = response.data.device
  
          }) 
      },
    closeModal() {
         this.showModal = false;
          },

      android() {
        axios
          .get("android/")
          .then((response) => {
            console.log(response.data);
            this.CPU_Use_And = parseFloat(response.data.CPU_Use).toFixed(3)
            this.DEVICE_name_And = response.data.OS_Version
            this.MEM_Use_And = parseFloat(response.data.MEM_Use).toFixed(3)
            this.DISK_Free_And = response.data.DISK_Free
            this.GPU_Use_And = parseFloat(response.data.GPU_Use).toFixed(3)
  
          })
      },
      mcu() {
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
      }
  
    },
  
  
  }
  
  
  
  
  </script>
  
  <style scoped>
  .head-font {
    font-weight: bolder;
    font-size: 30px;
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
  
  #dog {
    border-radius: 50px;
    height: 50px;
    width: 50px;
    filter: grayscale(0%);
  }
  
  #head .ivu-form-item-label {
    font-weight: bolder;
    font-size: 24px;
  }

  .modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .modal-content {
    background-color: white;
    padding: 20px;
    border-radius: 4px;
  }

  </style>
  