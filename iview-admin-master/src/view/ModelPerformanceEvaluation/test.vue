<template>
    <div>
      <label>
        <input type="checkbox" v-model="selectedValuePerformanceMulti" value="MaxAcc" />
        精确度最高
      </label>
  
      <input type="number" v-model="myNumber" :disabled="!selectedValuePerformanceMulti" />
      <p>输入的数字是: {{ myNumber }}</p>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        selectedValuePerformanceMulti: false,
        myNumber: 0
      };
    },
    watch: {
      selectedValuePerformanceMulti(newValue) {
        if (!newValue) {
          this.myNumber = 0;
        }
      }
    }
  };
  </script>
  