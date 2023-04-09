<template>
  <div>
    <label>
      <input type="radio" v-model="selectedValue" value="option1"> Option 1
    </label>
    <label>
      <input type="radio" v-model="selectedValue" value="option2"> Option 2
    </label>
    <label>
      <input type="radio" v-model="selectedValue" value="option3"> Option 3
    </label>
  </div>
</template>

<script>
export default {
  data() {
    return {
      selectedValue: ''
    };
  }
};
</script>
