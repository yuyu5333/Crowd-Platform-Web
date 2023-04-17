<template>
  <div>
    <button @click="showPopup">显示弹窗</button>
    <div v-if="show" class="popup">
      <h2>弹窗</h2>
      <compress-component />
      <button @click="hidePopup">关闭</button>
    </div>
    <div v-if="show" class="overlay" @click="hidePopup"></div>
  </div>
</template>

<script>
import CompressComponent from "./CompressComponent.vue";

export default {
  components: {
    CompressComponent
  },
  data() {
    return {
      show: false
    };
  },
  methods: {
    showPopup() {
      this.show = true;
    },
    hidePopup() {
      this.show = false;
    }
  }
};
</script>

<style>
.popup {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: white;
  padding: 20px;
  z-index: 10;
}
.overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 5;
}
</style>
