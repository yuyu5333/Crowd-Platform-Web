<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<!-- <script>
export default {
  name: 'App'
}
</script> -->

<script>
import { provide } from "vue"
import * as echarts from "echarts"
export default {
  name: 'App',
  setup(){
    provide("echarts", echarts)
  }
}
</script>

<style lang="less">
.size{
  width: 100%;
  height: 100%;
}
html,body{
  .size;
  overflow: hidden;
  margin: 0;
  padding: 0;
}
#app {
  .size;
}
</style>
