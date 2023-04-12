<template>
  <div>
    <div>
      <!-- 数据集选择start -->
      <card shadow>
        <Row>
          <div class="image-grid">
            <div class="image-class">
              <!-- <p>图像分类</p> -->
              <label>
                <input type="radio" v-model="selectedValueClass" value="图像分类" />
                图像分类
              </label>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" @click="selectedValueDataSet = 'Ade20k'" alt="Image 1" />
              <p class="image-info" @click="selectedValueDataSet = 'Ade20k'">Ade20k</p>
            </div>
            <div class="image-item">
              <img class="hover-animation" src="/images/dataset_images/CIFAR100.jpg" @click="selectedValueDataSet = 'Cifar100'" alt="Image 5" />
              <p class="image-info" @click="selectedValueDataSet = 'Cifar100'">Cifar100</p>
            </div>
            <div class="image-item">
              <img class="hover-animation" src="/images/dataset_images/VOC2012.jpg" @click="selectedValueDataSet = 'PascalVOC'" alt="Image 4" />
              <p class="image-info" @click="selectedValueDataSet = 'PascalVOC'">PascalVOC</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Caltech101.png" @click="selectedValueDataSet = 'Caltech101'" alt="Image 2" />
              <p class="image-info" @click="selectedValueDataSet = 'Caltech101'">Caltech101</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/minist.jpg" alt="Image 1" />
              <p class="image-info">Mnist</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Caltech256.jpg" alt="Image 3" />
              <p class="image-info">Caltech256</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/COCO.png" alt="Image 7" />
              <p class="image-info">Coco</p>
            </div>
            <div class="image-item">
              <img
                src="/images/dataset_images/CUB-200-2011.jpeg"
                alt="Image 6"
              />
              <p class="image-info">Cub200</p>
            </div>
          </div>
          <!-- 图像分类 -->
          <div class="image-grid">
            <div class="image-class">
              <div class="four-space">&nbsp;</div>
              <div>&nbsp;&nbsp;</div>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Food-101.jpg" alt="Image 2" />
              <p class="image-info">Food101</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/CIFAR100.jpg" alt="Image 3" />
              <p class="image-info">Hico-det</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Cityscapes.jpg" alt="Image 4" />
              <p class="image-info">ImageNet</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/COCO.png" alt="Image 5" />
              <p class="image-info">OxfordFlowers</p>
            </div>
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
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">ADE20K</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Cityscapes</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">ImageNet</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/CIFAR10.png" alt="Image 2" />
              <p class="image-info">KITTI</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/CIFAR100.jpg" alt="Image 3" />
              <p class="image-info">PascalVOC</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Cityscapes.jpg" alt="Image 4" />
              <p class="image-info">Cityscapes</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/COCO.png" alt="Image 5" />
              <p class="image-info">Coco</p>
            </div>
            <div class="image-item">
              <img
                src="/images/dataset_images/CUB-200-2011.jpeg"
                alt="Image 6"
              />
              <p class="image-info">Cub200</p>
            </div>
          </div>
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
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Coco</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Hico-det</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">ImageNet</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/CIFAR10.png" alt="Image 2" />
              <p class="image-info">IndoorSceneRecognition</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/CIFAR100.jpg" alt="Image 3" />
              <p class="image-info">Kitti</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/Cityscapes.jpg" alt="Image 4" />
              <p class="image-info">OxfordFlowers</p>
            </div>
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
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Hmdb51</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Ucf101</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Davis</p>
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
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Kinetics</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Sports-1M</p>
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
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">BAbI</p>
            </div>
            <div class="image-item">
              <img src="/images/dataset_images/ADE20K.jpg" alt="Image 1" />
              <p class="image-info">Snli</p>
            </div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
            <div class="image-item"></div>
          </div>
        </Row>
      </card>
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
  margin-top: 15px;
  font-size: 17px;
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
