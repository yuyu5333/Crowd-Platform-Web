<template>
  <div class="image-container">
    <div class="row">
      <div class="column" @click="gotoPage1">
        <img src="/images/fenge.png" alt="Image 1" />
      </div>
      <div class="column" @click="gotoPage2">
        <img src="/images/yasuo.png" alt="Image 2" />
      </div>
    </div>
    <div class="row">
      <div class="column" @click="gotoPage3">
        <img src="/images/qianrushi.png" alt="Image 3" />
      </div>
      <div class="column" @click="gotoPage4">
        <img src="/images/qianghuaxuexi.jpg" alt="Image 4" />
      </div>
    </div>

  </div>





</template>

<script>
export default {
  methods: {
    gotoPage1() {
      // 跳转到图片1的页面
      this.$router.push({ name: "cog_natural" });
    },
    gotoPage2() {
      // 跳转到图片2的页面
      this.$router.push({ path: "/ModelPerformaceEvaluation/firstpage" });
    },
    gotoPage3() {
      // 跳转到图片3的页面
      this.$router.push({ name: "ModelCompress" });
    },
    gotoPage4() {
      // 跳转到图片4的页面
      this.$router.push({ name: "segmentation" });
    },
  },
};
</script>

<style scoped>
.image-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}
.image {
  width: 25%;
  height: 25%;
  margin: 1rem;
  cursor: pointer;
}

.row {
  display: flex;
  width: 100%;
}

.column {
  flex-basis: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}

img {
  width: 80%;
  height: auto;
  cursor: pointer;
}
</style>


      <router-link tag="div" class="image" to="/cog_natural/cog_natural">
            <img src="/images/fenge.png" alt="Image 1" class="hover-animation" 
            @mouseover="showDescription = true" 
            @mouseleave="showDescription = false"
            v-b-tooltip.hover.top="datatest"
            />
          </router-link>
          <div v-if="showDescription" class="description">
          <p>这是图片的说明</p></div>