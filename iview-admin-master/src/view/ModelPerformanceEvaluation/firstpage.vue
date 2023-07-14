<template>
  <div>
    <!-- 预定义模型卡片 -->
    <Card shadow title="预定义模型">
      <row class="model-first-page" :gutter="32">
        <!-- 预定义模型介绍 -->
        <i-col span="32">
          <div class="model-system-text">
            <p>
              采用系统提供的深度学习模型。针对图像、视频处理领域多种深度学习模型的计算量、
              参数量、存储量、时延、能耗、精度性能评估。包括AlexNet、MobileNet、ResNet、VGG
              基础网络及其变体，供用户参考和下载使用
            </p>
          </div>
        </i-col>
      </row>
    </Card>

    <!-- 自定义模型卡片 -->
    <Card shadow title="自定义模型" style="margin-top: 10px">
      <row class="model-first-page" :gutter="32">
        <!-- 预定义模型介绍 -->
        <i-col span="32">
          <div class="model-system-text">
            <p>
              用户上传深度学习模型进行评估。根据《模型定义规范》构建用户的深度学习模型，
              上传模型文件后将提供计算量、参数量、存储量、时延、能耗、精度性能评估，供用户参考评估
            </p>
          </div>
          <Row style="margin-top: 10px">
            <!-- <Button
              icon="md-download"
              :loading="exportLoading"
            >
                <a 
              :href=
              "
              'api/download-modeldefinition/?modeldefinition=Modeldefinition'
              " 
              >
              点击下载《模型定义规范》
              </a>
            </Button> -->
            
            <Button type="primary" @click="openModal">点击下载《模型定义规范》</Button>

          </Row>
          <Row style="margin-top: 10px">
            <Button
              class="back-btn" @click="goBack"
            >
              返回首页
            </Button>
            
          </Row>
        </i-col>
        
      </row>
    
    
      <div id="myModal" class="modal">
          <div class="modal-content">
          <Row>
          <img class="modal-image" src="../../assets/images/contact.png" alt="弹窗图片">
              <span class="close">&times;</span>
          </Row>
          <Row>
              <button class="close-button" @click="closeModal">关闭</button>
          </Row>
          </div>
      </div> 
    </Card>

    <!-- 模型定义规范 -->
    <!-- <Card shadow title="模型定义规范" style="margin-top: 10px">
      <row class="info-title">
        <h3>一、头文件</h3>
        <div class="info-text">
          目前仅支持Pytorch架构的网络模型，需要添加以下头文件：pytorch环境
        </div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code1"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>
        <div class="info-text">
          注意：除此之外，符合Pytorch官方函数库的网络定义均可以使用，例如：
        </div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code2"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>

        <h3>二、网络定义</h3>
        <div class="info-text">
          符合Pytorch的网络结构定义均可作为自定义网络结构上传来进行模型性能评估
        </div>
        <div class="info-text">
          例如：AlexNet定义为Model_user, 其文件名为Model_user.py
        </div>
        <div class="info-text">
          注意：除此之外，符合Pytorch官方函数库的网络定义均可以使用，例如：
        </div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code3"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>

        <h3>三、测试函数</h3>
        <div class="info-text">
          首先需要用户验证函数定义是否正确，以Model_user为例：
        </div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code4"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>
        <div class="info-text">输出应该和模型定义输出大小相等：</div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code5"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>

        <h3>四、模型返回函数</h3>
        <div class="info-text">
          用于网络性能评估，以AlexNet网络为例，需要定义模型、输入张量以及模型的名称：
        </div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code6"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>
        <div class="info-text">
          注意：在模型定义时，需要先使用test()函数通过函数测试，再定义model_user()超参数函数
        </div>

        <h3>五、使用方法(待更新)</h3>
        <div class="info-text">
          主函数接口为"User_defined_model"和"--Computation True"：
        </div>
        <div>
          <prism-editor
            class="my-editor height-300"
            v-model="code7"
            :highlight="highlighter"
            :line-numbers="lineNumbers"
            :readonly="readonlyType"
          ></prism-editor>
        </div>
      </row>
    </Card> -->


  </div>
</template>

<script>
import { PrismEditor } from "vue-prism-editor";
import "vue-prism-editor/dist/prismeditor.min.css"; // import the styles somewhere
// import highlighting library (you can use any library you want just return html string)
import { highlight, languages } from "prismjs/components/prism-core";
import "prismjs/components/prism-clike";
import "prismjs/components/prism-javascript";
import "prismjs/themes/prism-tomorrow.css"; // import syntax highlighting styles
export default {
  name: "model-first-page",
  data() {
    return {
      code1: "import torch\n" + "import torch.nn as nn",
      code2: "import torch.nn.functional as F",

      code3:
        "class Model_user(nn.Module):\n" +
        "    def __init__(self):\n" +
        "        super(Model_user, self).__init__()\n" +
        "        self.Conv2d_1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=96, padding=1)\n" +
        "        self.bn_1 = nn.BatchNorm2d(96)\n" +
        "        self.maxpool_1 = nn.MaxPool2d((3, 3), stride=2, padding=1)\n\n" +
        "        self.Conv2d_2 = nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=2)\n" +
        "        self.bn_2 = nn.BatchNorm2d(256)\n" +
        "        self.maxpool_2 = nn.MaxPool2d((3, 3), stride=2, padding=1)\n\n" +
        "        self.Conv2d_3 = nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=1)\n" +
        "        self.Conv2d_4 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=1)\n" +
        "        self.Conv2d_5 = nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, padding=1)\n" +
        "        self.bn_3 = nn.BatchNorm2d(256)\n" +
        "        self.maxpool_3 = nn.MaxPool2d((3, 3), stride=2, padding=1)\n\n" +
        "        self.fc_1 = nn.Linear(4*4*256, 2048)\n" +
        "        self.dp_1 = nn.Dropout()\n" +
        "        self.fc_2 = nn.Linear(2048, 1024)\n" +
        "        self.dp_2 = nn.Dropout()\n" +
        "        self.fc_3 = nn.Linear(1024, 10)\n\n" +
        "    def forward(self, x):\n" +
        "        x = self.Conv2d_1(x)\n" +
        "        x = self.bn_1(x)\n" +
        "        x = F.relu(x)\n" +
        "        x = self.maxpool_1(x)\n\n" +
        "        x = self.Conv2d_2(x)\n" +
        "        x = self.bn_2(x)\n" +
        "        x = F.relu(x)\n" +
        "        x = self.maxpool_2(x)\n\n" +
        "        x = F.relu(self.Conv2d_3(x))\n" +
        "        x = F.relu(self.Conv2d_4(x))\n" +
        "        x = F.relu(self.Conv2d_5(x))\n" +
        "        x = self.bn_3(x)\n" +
        "        x = F.relu(x)\n" +
        "        x = self.maxpool_3(x)\n\n" +
        "        x = x.view(-1, 4*4*256)\n" +
        "        x = F.relu(self.fc_1(x))\n" +
        "        x = self.dp_1(x)\n" +
        "        x = F.relu(self.fc_2(x))\n" +
        "        x = self.dp_2(x)\n" +
        "        x = self.fc_3(x)\n" +
        "        return x\n",

      code4:
        "def test():\n" +
        "    net = Model_user()\n" +
        "    x = torch.randn(2, 3, 32, 32)\n" +
        "    y = net(x)\n" +
        "    print(y.size())",

      code5: "torch.Size([2, 10])",

      code6:
        '# 注意定义返回函数时，需要定义为"model_user"\n' +
        "def model_user():\n" +
        "    # 定义模型和输入\n" +
        "    model = AlexNet()\n" +
        "    input = torch.randn(2, 3, 32, 32)\n" +
        "    # 返回模型和输入\n" +
        "    return model, input",

      code7: "python Model_evaluation.py User_defined_model --Computation True",

      lineNumbers: true, // true 显示行号   false 不显示行号
      readonlyType: true, //true不可编辑   false 可编辑
    };
  },
  components: {
    PrismEditor,
  },
  mounted() {},
  watch: {},
  methods: {
    highlighter(code) {
      return highlight(code, languages.js); //returns html
    },
    goBack() {
      this.$router.push('/')
    },
    openModal() {
        document.getElementById('myModal').style.display = 'block';
      },
    closeModal() {
        document.getElementById('myModal').style.display = 'none';
      }
  },
};
</script>
<style>
@import "./common.less";
.model-first-page {
  text-align: center;
  font-size: 20px;
}
.model-first-page-other-icon {
  width: 20px;
  vertical-align: middle;
  margin-right: 6px;
}
.model-system-text {
  text-align: middle;
  margin-right: 6px;
  vertical-align: middle;
}
.model-first-page-other {
  text-align: left;
}
.model-first-page-other .ivu-btn {
  margin-right: 6px;
}
.info-title {
  margin-left: 10px;
}
.info-text {
  margin-left: 30px;
}
.my-editor {
  background: #2d2d2d;
  color: #ccc;
  font-family: Fira code, Fira Mono, Consolas, Menlo, Courier, monospace;
  font-size: 14px;
  line-height: 1.5;
  padding: 5px;
}
.prism-editor__textarea:focus {
  outline: none;
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

.modal {

display: none;
position: fixed;
z-index: 9999;
left: 5%;
top: 25%;
width: 100%;
height: 100%;
background-color: rgba(255, 255, 255, );
overflow: auto;
}

.modal-content {
position: relative;
background-color: #fff;
margin: auto;
padding: 20px;
width: 80%;
max-width: 600px;
text-align: center;
}
.modal-image {
max-width: 100%;
max-height: 100%;
}
.close {
position: absolute;
top: 10px;
right: 10px;
font-size: 20px;
font-weight: bold;
color: #000;
cursor: pointer;
}

</style>
