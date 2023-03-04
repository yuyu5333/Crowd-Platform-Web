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
        <Divider style="margin: 12px 0px"/>
        <Row>
          <Col span="6">设备名:
            <p class="text-font">{{ OS_Version_rasp }}</p>
          </Col>
          <Col span="6">CPU_Use:
            <p class="text-font">{{ CPU_Use_rasp }}</p>
          </Col>
          <Col span="6">MEM_Use:
            <p class="text-font">{{ MEM_Use_rasp }}</p>
          </Col>
          <Col span="6">DISK_Free:
            <p class="text-font">{{ DISK_Free_rasp }}</p>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px"/>
        <Row>
          <Col span="6">设备名:
            <p class="text-font">{{ DEVICE_name_JET }}</p>
          </Col>
          <Col span="6">CPU_Use:
            <p class="text-font">{{ CPU_Use_JET }}</p>
          </Col>
          <Col span="6">GPU_Use:
            <p class="text-font">{{ GPU_Use_JET }}</p>
          </Col>
          <Col span="6">MEM_Use:
            <p class="text-font">{{ MEM_Use_JET }}</p>
          </Col>
        </Row>
      </card>
      <div>
        <br>
      </div>
      <!-- <card shadow>
        <Row>
          <Col span="3">
            <p class="head-font">剩余设备：</p>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px"/>
        <Row>
          <Col span="6">设备名:
            <p class="text-font">{{ deviceStatus.DeviceName }}</p>
          </Col>
          <Col span="6">GPU:
            <p class="text-font">{{ GPU_Use }}</p>
          </Col>
          <Col span="6">CPU:
            <p class="text-font">{{ CPU_Use }}</p>
          </Col>
          <Col span="6">DRAM(GB):
            <p class="text-font">{{ MEM_Use }}</p>
          </Col>
        </Row>
        <Divider style="margin: 12px 0px"/>
        <Row>
          <Col span="6">设备名:
            <p class="text-font">{{ deviceStatus.DeviceName }}</p>
          </Col>
          <Col span="6">GPU:
            <p class="text-font">{{ deviceStatus.BatteryVolume }}</p>
          </Col>
          <Col span="6">CPU:
            <p class="text-font">{{ deviceStatus.CPU }}</p>
          </Col>
          <Col span="6">DRAM(GB):
            <p class="text-font">{{ deviceStatus.DRam }}</p>
          </Col>
        </Row>
      </card> -->
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
      <Divider style="margin: 12px 0px"/>
      <Form model="formItem" :label-width="60">
        <FormItem label="设备选择" class="head-font">
          <RadioGroup v-model="formItem.任务分类" size="large">
              <Radio label="任务1">设备1</Radio>
              <Radio label="任务2">设备2</Radio>
          </RadioGroup>
      </FormItem>
        <FormItem label="任务分类" class="head-font">
            <RadioGroup v-model="formItem.任务分类" size="large">
                <Radio label="任务1">图像分类</Radio>
                <Radio label="任务2">目标检测</Radio>
            </RadioGroup>
        </FormItem>
        <FormItem label="系统模型" class="head-font">
          <RadioGroup v-model="formItem.系统模型" size="large">
              <Radio label="模型1">AlexNet</Radio>
              <Radio label="模型2">MobileNet</Radio>
              <Radio label="模型3">ResNet</Radio>
          </RadioGroup>
      </FormItem>
      <Row>
        <Col span="2" ><b>自定义模型:</b>
        </Col>
        <Upload action="//jsonplaceholder.typicode.com/posts/">
            <Button icon="ios-cloud-upload-outline" >上传文件</Button>
        </Upload>
      </Row>
        <Button type="primary">感知模型</Button>
        <Button type="primary" style="margin-left: 33px">模型推理</Button>
      </Form>
    </card>
    <div>
      <br>
    </div>
    <card shadow>
      <Row>
        <Col span="3">
          <p class="head-font">推理结果：</p>
        </Col>
      </Row>
      <Divider style="margin: 12px 0px"/>
      <Row>
        <Col span="6">模型名称:
          <p class="text-font">{{ deviceStatus.DeviceName }}</p>
        </Col>
        <Col span="6">分割策略:
          <p class="text-font">{{ deviceStatus.DeviceName }}</p>
        </Col>
        <Col span="6">推理精度:
          <p class="text-font">{{ deviceStatus.BatteryVolume }}</p>
        </Col>
        <Col span="6">推理时间:
          <p class="text-font">{{ deviceStatus.CPU }}</p>
        </Col>
      </Row>
    </card>
  </div>
</template>
<script>
import axios from "axios";

export default {
  data() {
    return {
      mission: [],
      data: [{
        value: 'image_classification',
        label: '图像分类',
        children: [
          {
            value: 'Vgg16',
            label: 'Vgg16'
          },
          {
            value: 'AlexNet',
            label: 'AlexNet'
          }
        ]
      }, {
        value: 'semantic_segmentation',
        label: '语义分割',
        disabled: true,
        children: []
      }],
      deviceList: [
        {
          value: 'Xiaomi 12',
          label: 'Xiaomi 12'
        },
        {
          value: 'Iphone 14',
          label: 'Iphone 14'
        }
      ],
      device: [],
      compressRate: 75,
      deviceStatus: [],
      missionStatus: [],
      compressStatus: [],
      CPU_Use_rasp: "",
      OS_Version_rasp : "",
      RAM_Total_rasp : 0.0,
      MEM_Use_rasp: 0.0,
      CPU_Arch_rasp: "",
      DISK_Free_rasp: "10",

      DEVICE_name_JET:"",
      CPU_Use_JET:"",
      GPU_Use_JET:"",
      MEM_Use_JET:"",
      DISK_Free_JET:"50",
      
      formItem: {
                任务类型: "",
                模型选择: ""
            }
    }
  },
  mounted() {
    this.timer = setInterval(() => {
        this.addData()
        this.addData_jetson()
      },1000
    )
  },
  watch: {},

  methods: {
    // 3.发送axios无参数请求
    addData() {
      axios
        // 3.1url地址
        .get("/socket/")
        // 3.2成功时回调函数
        .then((response) => {
          console.log(response);
          // this.CPU_Use.push(parseFloat(response.data.CPU_Use));
          // this.DISK_Free.push(parseFloat(response.data.DISK_Free));
          // this.GPU_Use.push(parseFloat(response.data.GPU_Use).toFixed(3));
          this.CPU_Use_rasp = response.data.CPU_Use
          this.OS_Version_rasp = response.data.OS_Version
          this.RAM_Total_rasp  = response.data.RAM_Total 
          this.MEM_Use_rasp = response.data.MEM_Use
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
        .get("/jetson/")
        .then((response) => {
          console.log(response);
          this.CPU_Use_JET = response.data.CPU_Use 
          this.DEVICE_name_JET = response.data.DEVICE_NAME
          this.MEM_Use_JET = response.data.MEM_Use
          this.DISK_Free_JET = response.data.DISK_Free 
          this.GPU_Use_JET = response.data.GPU_Use 
        })   
    }
  }
}
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
</style>
