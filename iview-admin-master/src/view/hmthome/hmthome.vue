<template>
  <div
    class="background-image"
    :style="{ backgroundImage: 'url(' + backgroundImageUrl + ')' }"
  >
    <!-- <div> -->
    <div class="image-container">
      <row class="row">
        <router-link tag="div" class="image" to="/cog_natural/cog_natural">
          <img
            src="/images/ziyuanganzhi.png"
            alt="Image 1"
            class="hover-animation"
            v-b-tooltip.hover.top="imageDescription.deviceinfo"
          />
        </router-link>
        <div class="text">
          <h2 class="bold-header">
            <a
              href="http://blog.crowdhmt.com/articles/embedded_Intelligence"
              target="_blank"
              >设备资源感知</a
            >
          </h2>
          <!-- <p>支持为用户上传模型定制模型压缩方案，缩减模型体量以满足设定资源</p> -->
          <p></p>

          <p>支持于多种智能设备以及资源受限平台上(支持MCU、树莓派、FPGA、</p>
          <p>GPUnano、手机5种设备实时资源感知)</p>

        </div>
      </row>

      <row class="row">
        <router-link
          tag="div"
          class="image"
          to="/ModelPerformaceEvaluation/firstpage"
        >
          <img
            src="/images/moxingganzhi.png"
            alt="Image 2"
            class="hover-animation"
            v-b-tooltip.hover.top="imageDescription.modelinfo"
          />
        </router-link>
        <div class="text">
          <h2 class="bold-header">
            <a
              href="http://blog.crowdhmt.com/articles/mechanism"
              target="_blank"
              >模型性能评估</a
            >
          </h2>
          <!-- <p>支持为用户上传模型定制模型压缩方案，缩减模型体量以满足设定资源</p> -->
          <p></p>
          <p>支持用户自定义模型上传，评估硬件相关性能（能耗、时延），权重相</p>
          <p>关性能（精度），模型结构相关性能（计算量、存储量）</p>

          
        </div>
      </row>

      <row class="row">
        <router-link tag="div" class="image" to="/ModelCompress/ModelCompress">
          <img
            src="/images/yasuo.png"
            alt="Image 3"
            class="hover-animation"
            v-b-tooltip.hover.top="imageDescription.modelcompress"
          />
        </router-link>
        <div class="text">
          <h2 class="bold-header">
            <a
              href="http://blog.crowdhmt.com/articles/model_compression"
              target="_blank"
              >端自适应模型压缩</a
            >
          </h2>
          <p></p>
          <p>支持为用户上传模型定制模型压缩方案，缩减模型体量以满足设定资源</p>
          <p>为了在计算资源有限的设备上部署模型，例如移动设备和嵌入式系统等</p>
        </div>
      </row>

      <row class="row">
        <router-link tag="div" class="image" to="/segmentation/segmentation">
          <img
            src="/images/fenge.png"
            alt="Image 3"
            class="hover-animation"
            v-b-tooltip.hover.top="imageDescription.modelcompress"
          />
        </router-link>
        <div class="text">
          <h2 class="bold-header">
            <a
              href="http://blog.crowdhmt.com/articles/model_partition"
              target="_blank"
              >端自适应模型分割</a
            >
          </h2>
          <p></p>
     <!-- <p>支持为用户上传模型定制模型压缩方案，缩减模型体量以满足设定资源</p> -->
          <p>支持将一个大型的深度学习模型划分成多个部分，并将这些部分分别部</p>
          <p>署到不同的设备上进行计算，从而提高深度学习模型的性能，并降低运</p>
          <p>行成本</p>
        </div>
      </row>

    </div>
  </div>
</template>

<script>
import { VBTooltip } from "bootstrap-vue";
import "bootstrap/dist/css/bootstrap.css";
import "bootstrap-vue/dist/bootstrap-vue.css";

export default {
  data() {
    return {
      showDescription: false,
      imageDescription: {
        deviceinfo: "设备资源感知: 对您的智能设备可用资源进行测评",
        modelinfo: "模型性能评估: 对您的深度学习模型性能进行评估",
        modelcompress:
          "模型压缩: 减小模型大小以及减少模型参数数量，从而提高模型的效率和性能",
        mdoelseg:
          "模型分割: 将深度学习模型划分成若干子模型，以便在多个设备运行，从而减少计算和存储资源的需求",
      },
      // backgroundImageUrl: '/images/background.jpeg'
      backgroundImageUrl: "/images/background2.jpg",
      // backgroundImageUrl: "/images/background4.png",
    };
  },
  directives: { "b-tooltip": VBTooltip },
};
</script>


<style>
.background-image {
  background-size: cover;
  /* background-position: center center; */
  background-repeat: no-repeat;
  height: 140vh;
  background-attachment: fixed;
  opacity: 0.95;
}

.image-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  justify-content: left;
  align-items: center;
  height: 140vh;
  width: 200vh;

  margin: 0 auto;
  overflow: hidden; /* 清除浮动 */
}

.image-container .image {
  order: 1;
  float: left;
  width: 500px;
  margin-right: 20px;
}

.image-container .text {
  order: 2;
  float: right;
  overflow: hidden;
  font-weight: bold;
  color: white;
  font-size: 20px;
}

.bold-header {
  font-weight: bold;
  color: rgb(255, 0, 55);
}

.row {
  display: flex;
  width: 80%;
  justify-content: center;
  align-items: center;
}
img {
  width: 100%;
  height: auto;
  cursor: pointer;
}
.hover-animation {
  transition: transform 0.5s;
  
}

.hover-animation:hover {
  transform: scale(1.1);
}



</style>
