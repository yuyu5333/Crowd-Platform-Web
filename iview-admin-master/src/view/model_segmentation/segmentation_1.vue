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
          <Col span="6">MEM_Free:
          <p class="text-font">{{ MEM_Use_And }}GB</p>
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
            <Checkbox label="option3">设备3</Checkbox>
          </CheckboxGroup>
        </FormItem>
        <FormItem label="系统模型" class="head-font">
          <RadioGroup v-model="formItem.model" size="large">
            <Radio label="AlexNet">AlexNet</Radio>
            <Radio label="MobileNet">MobileNet</Radio>
            <Radio label="ResNet">ResNet</Radio>
            <Radio label="Customized">自定义模型</Radio>
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
            <Radio label="memory">内存最小</Radio>
            <Radio label="energy">能耗最小</Radio>
          </RadioGroup>
        </FormItem>
        <Row>
          <Col span="3" style="font-size: 24px;"><b>自定义模型: </b></Col>
          <Upload action="//jsonplaceholder.typicode.com/posts/" style="display: inline-block;">
            <Button icon="ios-cloud-upload-outline">上传模型</Button>
          </Upload>
          <Upload action="//jsonplaceholder.typicode.com/posts/" style="display: inline-block; margin-left:30px">
            <Button icon="ios-cloud-upload-outline">上传数据集</Button>
          </Upload>
        </Row>
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
        <p class="text-font">{{ deviceStatus.BatteryVolume }}</p>
        </Col>
        <Col span="6">分割策略:
        <p class="text-font">在第{{ seg_id }}层分割</p>
        </Col>
        <Col span="6">推理能耗:
        <p class="text-font">{{ deviceStatus.BatteryVolume }}</p>
        </Col>
        <Col span="6">推理时间:
        <p class="text-font">{{ seg_num }}</p>
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
      showflag: false,
      mission: [],
      data: [
        {
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
      device: [],
      compressRate: 75,
      deviceStatus: [],
      missionStatus: [],
      compressStatus: [],
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
      seg_num:"",

      CPU_Use_And :"",
      DEVICE_name_And:'',
      MEM_Use_And :'',
      DISK_Free_And:'',
      GPU_Use_And: "",

      formItem: {
        device: [],
        task: [],
        model: [],
        target: []
      }
    }
  },
  mounted() {
    this.timer = setInterval(() => {
        this.addData()
        this.addData_jetson()
        this.android()
      },1000
    )
  },
  watch: {},

  methods: {
    // 3.发送axios无参数请求
    addData() {
      axios
        // 3.1url地址
        .get("/raspberry/")
        // 3.2成功时回调函数
        .then((response) => {
          console.log(response);
          // this.CPU_Use.push(parseFloat(response.data.CPU_Use));
          // this.DISK_Free.push(parseFloat(response.data.DISK_Free));
          // this.GPU_Use.push(parseFloat(response.data.GPU_Use).toFixed(3));
          this.CPU_Use_rasp = response.data.CPU_Use
          this.OS_Version_rasp = response.data.OS_Version
          this.RAM_Total_rasp = response.data.RAM_Total
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
    },
    segmentation() {
      // this.showflag = true;
      // function addMargin() {
      //   var timestamp=new Date().getTime();
      //   let arrowImg = document.getElementById("arrowImg");
      //   let speed = 5;
      //   let leftMargin = timestamp%(650*speed)/speed;
      //   arrowImg.style.rotate = "1turn"
      //   arrowImg.style.paddingLeft = leftMargin + "px";
      // }
      axios
        .post("/segmentation/", {
          device: this.formItem.device,
          task: this.formItem.task,
          target: this.formItem.target,
          model: this.formItem.model
        })
        .then((response) => {
          console.log(response.data);
          this.seg_id = response.data.id
          this.seg_num = response.data.num

        })
      // setInterval(addMargin, 5);

    },

    android() {
      axios
        .get("/android/")
        .then((response) => {
          console.log(response.data);
          this.CPU_Use_And = parseFloat(response.data.CPU_Use).toFixed(3)
          this.DEVICE_name_And = response.data.OS_Version
          this.MEM_Use_And = parseFloat(response.data.MEM_Use).toFixed(3)
          this.DISK_Free_And = response.data.DISK_Free
          this.GPU_Use_And = parseFloat(response.data.GPU_Use).toFixed(3)

        })
      // setInterval(addMargin, 5);

    }

  },


}




</script>

<style>
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
</style>
