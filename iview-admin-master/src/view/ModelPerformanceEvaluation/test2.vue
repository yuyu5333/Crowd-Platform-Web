<template>
  <div class="rotate-div" @mouseenter="rotate">
    <p>鼠标触碰这个div会触发一次绕着div中心旋转的动画</p>
  </div>
</template>

<style>
.rotate-div {
  display: flex;
  justify-content: center;
  align-items: center;
  width: auto;
  height: auto;
  padding: 10px;
  border: 1px solid black;
  transition: transform 1s ease-in-out;
}

.rotate {
  animation: rotate 1s ease-in-out;
  animation-fill-mode: forwards;
  pointer-events: none;
}

@keyframes rotate {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>

<script>
export default {
  methods: {
    rotate() {
      const rotateDiv = document.querySelector('.rotate-div');
      if (!rotateDiv.classList.contains('rotate')) {
        rotateDiv.classList.add('rotate');
        setTimeout(() => {
          rotateDiv.classList.remove('rotate');
        }, 1000);
      }
    },
  },
};
</script>
