<template>
  <div class="container" @mouseover="animate">
    <div class="box" :style="{transform: `translate(${x}px, ${y}px)`}">
      <p>鼠标移入此处将会向右移动100像素</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      x: 0,
      y: 0,
    };
  },
  methods: {
    animate() {
      if (this.x < 100) {
        this.x += 100;
        this.$nextTick(() => {
          this.animate();
        });
      } else {
        this.x = 0;
      }
    },
  },
};
</script>

<style>
.container {
  width: 100%;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f0f0f0;
}

.box {
  display: flex;
  justify-content: center;
  align-items: center;
  width: auto;
  height: auto;
  padding: 20px;
  background-color: #fff;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
}
</style>
