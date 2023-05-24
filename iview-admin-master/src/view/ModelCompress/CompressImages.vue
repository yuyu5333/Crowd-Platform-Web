<template>
  <div>
    <div>
      <!-- 数据集选择start -->
        <Row>
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>图像分类别</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="图像分类" />
                图像分类
              </label>
            </div>
            <div class="image-item">
              <img class="hover-animation" src="/images/dataset_images/CIFAR100.jpg" @click="selectedValueDataSet = 'Cifar100'" alt="Image 5" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Cifar100'">Cifar100</p>
              <p class="image-info" @click="selectedValueDataSet = 'Cifar100'">50000张图像100类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Cifar100'">图像大小：32x32</p>

            </div>
            <div class="image-item">
              <img class="hover-animation" src="/images/dataset_images/VOC2012.jpg" @click="selectedValueDataSet = 'PascalVOC'" alt="Image 4" />
              <p class="image-info-name" @click="selectedValueDataSet = 'PascalVOC'">PascalVOC</p>
              <p class="image-info" @click="selectedValueDataSet = 'PascalVOC'">17000张图像20类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'PascalVOC'">图像大小：200-500</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Caltech101.png" @click="selectedValueDataSet = 'Caltech101'" alt="Image 2" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Caltech101'">Caltech101</p>
              <p class="image-info" @click="selectedValueDataSet = 'Caltech101'">9000张图像101类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Caltech101'">图像大小：200-500</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Caltech256.jpg" @click="selectedValueDataSet = 'Caltech256'" alt="Image 3" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Caltech256'">Caltech256</p>
              <p class="image-info" @click="selectedValueDataSet = 'Caltech256'">31000张图像256类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Caltech256'">图像大小：200-500</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/minist.jpg" @click="selectedValueDataSet = 'Mnist'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Mnist'">Mnist</p>
              <p class="image-info" @click="selectedValueDataSet = 'Mnist'">60000张图像10类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Mnist'">图像大小：28x28</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/COCO.png" @click="selectedValueDataSet = 'Coco'" alt="Image 7" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Coco'">Coco</p>
              <p class="image-info" @click="selectedValueDataSet = 'Coco'">33万张图像90类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Coco'">图像大小：480-1024</p>
            </div>
            <div class="image-item">
              <img class="hover-animation" src="/images/dataset_images/Food-101.jpg" @click="selectedValueDataSet = 'Food-101'" alt="Image 2" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Food-101'">Food-101</p>
              <p class="image-info" @click="selectedValueDataSet = 'Food-101'">10万张图像101类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Food-101'">图像大小：512x512</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/CUB-200-2011.jpeg" @click="selectedValueDataSet = 'Cub200'" alt="Image 6"/>
              <p class="image-info-name" @click="selectedValueDataSet = 'Cub200'">Cub200</p>
              <p class="image-info" @click="selectedValueDataSet = 'Cub200'">11000张图像200类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Cub200'">图像大小：600x600</p>
            </div>
          </div>
          <!-- 图像分类 -->
          <div class="image-grid">
            <div class="image-class">
              <div class="four-space">&nbsp;</div>
              <div>&nbsp;&nbsp;</div>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/imagenet.jpg" @click="selectedValueDataSet = 'ImageNet'" alt="Image 4" />
              <p class="image-info-name" @click="selectedValueDataSet = 'ImageNet'">ImageNet</p>
              <p class="image-info" @click="selectedValueDataSet = 'ImageNet'">1400万张图像1000类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'ImageNet'">多种图像大小</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/FLOWERS.jpg" @click="selectedValueDataSet = 'OxfordFlowers'" alt="Image 5" />
              <p class="image-info-name" @click="selectedValueDataSet = 'OxfordFlowers'">OxfordFlowers</p>
              <p class="image-info" @click="selectedValueDataSet = 'OxfordFlowers'">8189张图像102类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'OxfordFlowers'">图像大小：1024x768</p>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div>
          <!-- 语义分割 -->
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>语义分割</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="语义分割" />
                语义分割
              </label>
            </div>
            <div class="image-item">
              <img class="hover-animation" src="/images/dataset_images/ADE20K.jpg" @click="selectedValueDataSet = 'Ade20k'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Ade20k'">Ade20k</p>
              <p class="image-info" @click="selectedValueDataSet = 'Ade20k'">20000张图像150类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Ade20k'">多种图像大小</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Cityscapes.jpg" @click="selectedValueDataSet = 'Cityscapes'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Cityscapes'">Cityscapes</p>
              <p class="image-info" @click="selectedValueDataSet = 'Cityscapes'">5000张高质量像素级标注</p>
              <p class="image-info" @click="selectedValueDataSet = 'Cityscapes'">图像大小：2048x1024</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/KITTI.jpg" alt="Image 2" />
              <p class="image-info-name" @click="selectedValueDataSet = 'KITTI'">KITTI</p>
              <p class="image-info" @click="selectedValueDataSet = 'KITTI'">200张像素级标注</p>
              <p class="image-info" @click="selectedValueDataSet = 'KITTI'">图像大小：1226x370</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/PASCAL_VOC.jpg" @click="selectedValueDataSet = 'PascalVOC'" alt="Image 3" />
              <p class="image-info-name" @click="selectedValueDataSet = 'PascalVOC'">PascalVOC</p>
              <p class="image-info" @click="selectedValueDataSet = 'PascalVOC'">2913张标注21种类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'PascalVOC'">多种图像大小</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/FC_Dataset/C_Driving.jpg" @click="selectedValueDataSet = 'CDriving'" alt="Image 5" />
              <p class="image-info-name" @click="selectedValueDataSet = 'CDriving'">CDriving</p>
              <p class="image-info" @click="selectedValueDataSet = 'CDriving'">真实街道图像，包含多云、雨天、雪天等场景</p>
              <p class="image-info" @click="selectedValueDataSet = 'CDriving'">图像大小：1280x720</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/FC_Dataset/SHIFT.jpg" @click="selectedValueDataSet = 'Shift'" alt="Image 4" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Shift'">Shift</p>
              <p class="image-info" @click="selectedValueDataSet = 'Shift'">虚拟城市图像，包含15种不同天气场景</p>
              <p class="image-info" @click="selectedValueDataSet = 'Shift'">图像大小：1280x800</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/FC_Dataset/Synthia-Seq.png" @click="selectedValueDataSet = 'Synthia-Seq'" alt="Image 3" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Synthia-Seq'">Synthia-Seq</p>
              <p class="image-info" @click="selectedValueDataSet = 'Synthia-Seq'">虚拟城市场景，包含11种不同场景影响因素</p>
              <p class="image-info" @click="selectedValueDataSet = 'Synthia-Seq'">图像大小：1280x800</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/FC_Dataset/VirtualKITTI2.jpg" @click="selectedValueDataSet = 'VirtualKitti2'" alt="Image 2" />
              <p class="image-info-name" @click="selectedValueDataSet = 'VirtualKitti2'">VirtualKitti2</p>
              <p class="image-info" @click="selectedValueDataSet = 'VirtualKitti2'">汽车摄像图像，包含6种天气、光线干扰因素</p>
              <p class="image-info" @click="selectedValueDataSet = 'VirtualKitti2'">图像大小：1242x375</p>
            </div>
          </div>

          <!-- <div class="image-grid">
            <div class="image-class">
              <div class="four-space">&nbsp;</div>
              <div>&nbsp;&nbsp;</div>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div> -->

          <!-- 目标检测 -->
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>目标检测</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="目标检测" />
                目标检测
              </label>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/COCO.png" @click="selectedValueDataSet = 'Coco'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Coco'">Coco</p>
              <p class="image-info" @click="selectedValueDataSet = 'Coco'">12.2万张图像80类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Coco'">多种图像大小</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/HICO-DET.jpg" alt="Image 3" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Hico-det'">Hico-det</p>
              <p class="image-info" @click="selectedValueDataSet = 'Hico-det'">47776张图像80类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Hico-det'">图像大小：480-1280</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/imagenet.jpg" alt="Image 1" />
              <p class="image-info-name">ImageNet</p>
              <p class="image-info" @click="selectedValueDataSet = 'Hico-det'">45万张图像200类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Hico-det'">图像大小：480-1280</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/MITIndoor.jpg" @click="selectedValueDataSet = 'MITIndoor'" alt="Image 2" />
              <p class="image-info-name" @click="selectedValueDataSet = 'MITIndoor'">MITIndoor</p>
              <p class="image-info" @click="selectedValueDataSet = 'MITIndoor'">15620张图像67类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'MITIndoor'">图像大小：480-1280</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/TrafficSign.jpg" @click="selectedValueDataSet = 'TrafficSign'" alt="Image 3" />
              <p class="image-info-name" @click="selectedValueDataSet = 'TrafficSign'">TrafficSign</p>
              <p class="image-info" @click="selectedValueDataSet = 'TrafficSign'">2000张高清交通标志图像</p>
              <p class="image-info" @click="selectedValueDataSet = 'TrafficSign'">图像大小：1920x1080</p>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div>
          <!-- 行为识别 -->
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>行为识别</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="行为识别" />
                行为识别
              </label>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/HMDB51.jpg" @click="selectedValueDataSet = 'Hmdb51'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Hmdb51'">Hmdb51</p>
              <p class="image-info" @click="selectedValueDataSet = 'Hmdb51'">6766个视频，51个类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Hmdb51'">涵盖日常活动、体育运动和人类交互行为等</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/UCF101.jpg" @click="selectedValueDataSet = 'Ucf101'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Ucf101'">Ucf101</p>
              <p class="image-info" @click="selectedValueDataSet = 'Ucf101'">1.3万个视频，101个类别</p>
              <p class="image-info" @click="selectedValueDataSet = 'Ucf101'">涵盖各种运动、日常活动和体育运动，如篮球运动、刷牙、弹吉他等</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/DAVIS.png" @click="selectedValueDataSet = 'Davis'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Davis'">Davis</p>
              <p class="image-info" @click="selectedValueDataSet = 'Davis'">包含50个高质量视频序列</p>
              <p class="image-info" @click="selectedValueDataSet = 'Davis'">涵盖各种场景和挑战，如摄像机运动、遮挡、光照变化等</p>
            </div>
            <div class="image-item">
              <p class="image-info"></p>
            </div>
            <div class="image-item">
              <p class="image-info"></p>
            </div>
            <div class="image-item">
              <p class="image-info"></p>
            </div>
            <div class="image-item">
              <p class="image-info"></p>
            </div>
            <div class="image-item">
              <p class="image-info"></p>
            </div>
          </div>
          <!-- 动作检测 -->
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>动作检测</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="动作检测" />
                动作检测
              </label>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/kinetics.jpg" @click="selectedValueDataSet = 'Kinetics'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Kinetics'">Kinetics</p>
              <p class="image-info" @click="selectedValueDataSet = 'Kinetics'">30万个视频片段</p>
              <p class="image-info" @click="selectedValueDataSet = 'Kinetics'">涵盖了各种运动、日常活动和体育运动，如跳舞、做饭、打篮球等</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Sports-1M.png" @click="selectedValueDataSet = 'Sports-1M'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Sports-1M'">Sports-1M</p>
              <p class="image-info" @click="selectedValueDataSet = 'Sports-1M'">110万个YouTube视频片段</p>
              <p class="image-info" @click="selectedValueDataSet = 'Sports-1M'">包括各种体育运动、锻炼方式和技能，如篮球、足球、瑜伽、击剑等</p>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div>
          <!-- 语音文本 -->
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>语音文本</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="语音文本" />
                语音文本
              </label>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/BABI.png" @click="selectedValueDataSet = 'BAbI'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'BAbI'">BAbI</p>
              <p class="image-info" @click="selectedValueDataSet = 'BAbI'">包含20个不同的任务</p>
              <p class="image-info" @click="selectedValueDataSet = 'BAbI'">每个任务专注于一个特定的推理技能，如基本的事实回答、逻辑推理、时间推理等</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Snli.png" @click="selectedValueDataSet = 'Snli'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Snli'">Snli</p>
              <p class="image-info" @click="selectedValueDataSet = 'Snli'">包含了大约57万个人工标注的英语句子对，分为三种关系类型：蕴含、矛盾和中立</p>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div>
          <!-- 图像处理 -->
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>图像处理</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="图像处理" />
                图像处理
              </label>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/LXC_Dataset/Row.png" @click="selectedValueDataSet = 'Row'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Row'">Row</p>
              <p class="image-info" @click="selectedValueDataSet = 'Row'">低照度数据集，用于低照度增强，包含动态、静态镜头、动态、静态场景四种类型</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/BSDS500.jpg" @click="selectedValueDataSet = 'Bsds500'" alt="Image 1" />
              <p class="image-info-name" @click="selectedValueDataSet = 'Bsds500'">Bsds500</p>
              <p class="image-info" @click="selectedValueDataSet = 'Bsds500'">包括500幅高质量彩色图像，用于图像分割和边缘检测。图像大小：321x481</p>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div>
        </Row>
      <!-- 数据集选择end -->
    </div>
  </div>
</template>

<script>
export default {
  props: {
    selectedValueClass: {
      type: String,
      default: "",
    },
    selectedValueDataSet: {
      type: String,
      default: "",
    },
  },
  data() {
    return {
      selectedValueClass: this.selectedValueClass,
      selectedValueDataSet: this.selectedValueDataSet,
    };
  },
  watch: {
    selectedValueClass(newValue) {
      this.$emit("update-selected-value-class", newValue);
    },
    selectedValueDataSet(newValue) {
      this.$emit("update-selected-value-dataset", newValue);
    },
  },
};
</script>


<style scoped>
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
  margin-top: 10px;
  font-size: 13px;
}

.image-info-name {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 15px;
  font-size: 15px;
  color:rgb(64, 92, 235);
}

img {
  vertical-align: middle;
  border-radius: 50%;
  width: 110px;
  height: 110px;
}

.four-space {
  letter-spacing: 4em; /* 设置字母间距为 1 个字宽 */
}

.hover-animation {
  transition: transform 0.5s;
}

.hover-animation:hover {
  transform: scale(1.2);
}

</style>
