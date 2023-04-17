<template>
  <div class="demo-split">
    <Split v-model="split1">
      <template #left>
        <div class="demo-split-pane">
          <!-- 计算量 start -->
          <card shadow>
            <Row>
              <Col span="15">
                <p class="head-font-parameter">计算量(M)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8">
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Computation }}
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 计算量 end 参数量 start -->
          <card shadow>
            <Row>
              <Col span="15">
                <p class="head-font-parameter">参数量(M)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8">
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Parameter }}
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 参数量 end 存储量 start -->
          <card shadow>
            <Row>
              <Col span="15">
                <p class="head-font-parameter">存储量(MB)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8">
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Storage }}
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 存储量 end 时延 start -->
          <card shadow>
            <Row>
              <Col span="3">
                <p class="head-font-parameter">时延(ms)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8" class="text-font-device">Windows：
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Windows }}
                </p>
              </Col>
              <Col span="8" class="text-font-device">Raspberry Pi 4B：
                <p class="text-font-parameter">
                  {{ sysmodelStatus.RaspberryPi4B }}
                </p>
              </Col>
              <Col span="8" class="text-font-device">Jetson Nx：
                <p class="text-font-parameter">
                  {{ sysmodelStatus.JetsonNx }}
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 时延 end 能耗 start -->
          <card shadow>
            <Row>
              <Col span="15">
                <p class="head-font-parameter">能耗(mJ)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8">
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Energy }}
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 能耗 end 精确度 start -->
          <card shadow>
            <Row>
              <Col span="15">
                <p class="head-font-parameter">精确度(%)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8">
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Accuracy }}
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 精确度 end -->
          <button class="back-btn" @click="goBackhome">
            返回首页
          </button>
        </div>
      </template>
      <!-- 系统模型 -->
      <template #right>
        <div>
          <template>
            <div class="demo-split-pane">
              <card shadow>
                <Row>
                  <Col span="6">
                    <p class="head-font-parameter">模型选择：</p>
                  </Col>
                  <Col span="6">
                    <Select
                      v-model="sysmodel"
                      @on-change="checkSysModelChange"
                      style="width: 100px; font-size: 18px"
                    >
                      <Option
                        v-for="item in modelList"
                        :value="item.value"
                        :key="item.value"
                        >{{ item.label }}</Option
                      >
                    </Select>
                  </Col>
                </Row>
                <Divider style="margin: 12px 0px" />
                
                <Row style="margin-top: 10px">
                  <Button
                    type="primary"
                    style="font-size: 18px"
                    class="model-evaluation"
                  >
                    <a
                      :href="
                        'download-sysmodel/?model=' +
                        sysmodelStatus.SysModelName
                      "
                      style="color: white"
                    >
                      下载模型
                    </a>
                  </Button>
                  <Button
                    type="primary"
                    style="margin-left: 10px; font-size: 18px"
                    class="model-evaluation"
                  >
                    <a
                      :href="
                        'download-sysmodelcode/?modelcode=' +
                        sysmodelStatus.SysModelName
                      "
                      style="color: white"
                    >
                      下载模型源代码
                    </a>
                  </Button>
                </Row>
              </card>
              <Divider style="margin: 10px 0px" />
              <card shadow>
                <Row>
                  <Col span="6">
                    <p class="head-font-parameter">模型简介</p>
                  </Col>
                </Row>
                <Divider style="margin: 12px 0px" />
                <Row>
                  <p class="text-font-information">
                    {{ sysmodelStatus.Infomation }}
                  </p>
                </Row>
                <br>
              </card>
            </div>
          </template>
        </div>
      </template>
      <!-- 系统模型 end -->
    </Split>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      modelList: [
        {
          value: "AlexNet",
          label: "AlexNet",
        },
        {
          value: "MobileNet",
          label: "MobileNet",
        },
        {
          value: "ResNet",
          label: "ResNet",
        },
        {
          value: "VGG",
          label: "VGG",
        },
      ],
      split1: 0.618,
      split2: 0.382,
      sysmodel: [],
      sysmodelStatus: [],
      sysdevice: [],
      sysdeviceStatus: [],
    };
  },
  mounted() {},
  watch: {},
  methods: {
    checkSysModelChange(sysmodel) {
      let that = this;
      axios
        .post("get-sysmodel/", {
          SysModelName: sysmodel,
        })
        .then((response) => {
            console.log(response.data);
            that.sysmodelStatus = response.data
        });
    },
    goBackhome() {
      this.$router.push('/')
    }
  },
};
</script>


<style scoped>
.demo-split {
  height: 5000px;
  border: 1px solid #ffffff;
}
.head-font-parameter {
  font-weight: bolder;
  font-size: 16px;
}
.text-font-parameter {
  font-weight: bold;
  font-size: 16px;
  color: #6b25d3;
}
.text-font-information {
  font-size: 16px;
}
.text-font-device {
    font-weight: bold;
  font-size: 16px;
  color: #0b5fdd;
}
.back-btn {
  display: inline-block;
  background-color: #007bff;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  cursor: pointer;
}

.back-btn:hover {
  background-color: #0069d9;
}
</style>