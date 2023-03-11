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
                  <!--  -->
                  {{ UserModelStatus.Computation }}
                </p>
              </Col>
            </Row>
            <!-- <br /> -->
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
                  {{ UserModelStatus.Parameter }}
                </p>
              </Col>
            </Row>
            <!-- <br /> -->
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
                  {{ UserModelStatus.Storage }}
                </p>
              </Col>
            </Row>
            <!-- <br /> -->
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
              <p class="text-font-parameter">
                {{ UserModelStatus.Latency }}
              </p>
              <!-- <Col span="8" class="text-font-device"
                >Windows：
                <p class="text-font-parameter">
                  {{ sysmodelStatus.Windows }}
                </p>
              </Col>
              <Col span="8" class="text-font-device"
                >Raspberry Pi 4B：
                <p class="text-font-parameter">
                  {{ sysmodelStatus.RaspberryPi4B }}
                </p>
              </Col>
              <Col span="8" class="text-font-device"
                >Jetson Nx：
                <p class="text-font-parameter">
                  {{ sysmodelStatus.JetsonNx }}
                </p>
              </Col> -->
            </Row>
            <!-- <br /> -->
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
            <!-- <Row>
              <Col span="8">
                <p class="text-font-parameter" >
                  {{ UserModelStatus.Energy }}
                </p>
              </Col>
            </Row> -->
            <Row>
              <Col span="8">
                <transition name="fade">
                  <p class="text-font-parameter">
                    {{ currentItem  }}
                  </p>
                </transition>
              </Col>
            </Row>
            <!-- <br /> -->
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
                  {{ UserModelStatus.Accuracy }}
                </p>
              </Col>
            </Row>
            <!-- <br /> -->
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 精确度 end -->
          <button class="back-btn" @click="goBackhome">
            返回首页
          </button>
        </div>
      </template>
      <template #right>
        <div class="demo-split-pane">
          <card shadow title="自定义模型" size="large">
            <row class="model-upload">
              <upload :before-upload="beforeUpload" action="upload-usermodel/">
                <Button icon="ios-cloud-upload-outline" style="margin-top: 10px"
                  >点击上传模型</Button
                >
              </upload>
              <!-- <row><p>只能上传Python文件</p></row> -->
            </row>
            <row class="model-upload" style="margin-top: 10px">
              <p style="margin-left: 0px">{{ CheckModel_Value.CheckStatus }}</p>
              <Button
                type="primary"
                @click="CheckModel()"
                style="margin-right: 8px"
                >模型检验</Button
              >
              <Button
                type="primary"
                style="margin-left: 8px"
                @click="getUserModelChange()"
                >模型评估</Button
              >
              <pulse-loader :loading="loading" color="#5cb85c" />
            </row>
            <row> </row>
          </card>
        </div>
        <div>
          <template>
            <div class="demo-split-pane">
              <card shadow title="模型总览" size="large">
                <Row>
                  <!-- <br /> -->
                  <!-- <br /> -->
                  <!-- <br /> -->
                  <!-- <br /> -->
                </Row>
              </card>
            </div>
          </template>
        </div>
      </template>
    </Split>
  </div>
</template>

<script>
import axios from "axios";
import PulseLoader from "vue-spinner/src/PulseLoader.vue";

export default {
  components: {
    PulseLoader,
  },
  data() {
    return {
      split1: 0.618,
      split2: 0.382,
      formItem: {
        系统模型: "",
      },
      CheckModel_Value: [],
      UserModelStatus: [],
      loading: false,
      index: 0,
      values: [],
    };
  },
  computed: {
    currentItem() {
      return this.values[this.index]
    },
  },
  created() {
    // 假设从后端获取到的数据保存在了this.UserModelStatus中
    this.values = [
      this.UserModelStatus.Energy,
      this.UserModelStatus.Cl,
      this.UserModelStatus.Ml,
      this.UserModelStatus.CacheRate,
    ]
    setInterval(() => {
      // 假设从后端获取到的数据更新到了this.UserModelStatus中
      this.values = [
        this.UserModelStatus.Energy,
        this.UserModelStatus.Cl,
        this.UserModelStatus.Ml,
        this.UserModelStatus.CacheRate,
      ]
      this.index = (this.index + 1) % this.values.length
    }, 1500)
  },
  methods: {
    getUserModelChange() {
      let that = this;
      this.loading = true;
      axios
        .post("get-usermodel/", {
          UserModelName: true,
        })
        .then((response) => {
          that.UserModelStatus = response.data;
          console.log(response.data);
          this.loading = false;
        });
    },
    CheckModel() {
      let that = this;
      axios
        .post("check-usermodel/", {
          CheckModelValue: "True",
        })
        .then((response) => {
          that.CheckModel_Value = response.data;
          console.log(that.CheckModel_Value);
        });
    },
    beforeUpload(file) {
      let nameSplit = file.name.split(".");
      let format = nameSplit[nameSplit.length - 1];
      if (format === "py") {
        return true;
      } else {
        this.$Notice.warning({
          title: "只能上传Python文件",
          desc: "只能上传Python文件，请重新上传",
        });
      }
      return false;
    },
    goBackhome() {
      this.$router.push('/hmthome/hmthome')
    }
  },
};
</script>
<style scoped>
.demo-split {
  height: 5000px;
  border: 1px solid #dcdee2;
}
.demo-split-pane {
  padding: 10px;
}
.demo-split-pane.no-padding {
  height: 300px;
  padding: 0;
}
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
.model-upload {
  text-align: center;
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

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}

</style>