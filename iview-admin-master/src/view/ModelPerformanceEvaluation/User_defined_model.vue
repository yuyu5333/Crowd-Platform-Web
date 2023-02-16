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
                </p>
              </Col>
            </Row>
            <br/>
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
                  <!-- {{ sysmodelStatus.Parameter }} -->
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
                  <!-- {{ sysmodelStatus.Storage }} -->
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
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 时延 end 能耗 start -->
          <card shadow>
            <Row>
              <Col span="15">
                <p class="head-font-parameter">能耗(J)</p>
              </Col>
            </Row>
            <Divider style="margin: 12px 0px" />
            <Row>
              <Col span="8">
                <p class="text-font-parameter">
                  <!-- {{ sysmodelStatus.Energy }} -->
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
                  <!-- {{ sysmodelStatus.Accuracy }} -->
                </p>
              </Col>
            </Row>
            <br>
          </card>
          <Divider style="margin: 10px 0px" />
          <!-- 精确度 end -->
        </div>
      </template>
      <template #right>
        <div class="demo-split-pane">
          <card shadow title="自定义模型" size="large">
            <Form model="formItem" :label-width="80">
              <row class="model-upload">
                <Upload action="//jsonplaceholder.typicode.com/posts/">
                  <Button icon="ios-cloud-upload-outline" style="margin: 10px"
                    >点击上传模型</Button
                  >
                </Upload>
              </row>
              <row class="model-upload" style="margin-top: 10px">
                <Button type="primary" style="margin-left: 8px"
                  >模型检验</Button
                >
                <Button type="primary" style="margin-left: 8px"
                  >模型评估</Button
                >
              </row>
            </Form>
          </card>
        </div>
        <div>
          <template>
            <div class="demo-split-pane">
              <card shadow title="模型总览" size="large">
                <Row>
                  <br>
                  <br>
                  <br>
                  <br>
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
export default {
  data() {
    return {
      split1: 0.618,
      split2: 0.382,
      formItem: {
        系统模型: "",
      },
    };
  },
};
</script>
<style>
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
  font-size: 18px;
}
.text-font-parameter {
  font-weight: bold;
  font-size: 18px;
  color: #6b25d3;
}
.text-font-information {
  font-size: 18px;
}
.text-font-device {
  font-weight: bold;
  font-size: 18px;
  color: #0b5fdd;
}
.model-upload {
  text-align: center;
}
</style>