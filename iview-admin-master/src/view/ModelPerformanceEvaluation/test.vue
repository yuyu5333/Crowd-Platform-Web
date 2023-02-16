<template>
  <div>
    <select id="modelselect" name="modelselect" onchange="selectModelChange()">
      <option>---请选择模型---</option>
      <option>AlexNet</option>
      <option>MobileNet</option>
      <option>ResNet</option>
      <option>VGG</option>
    </select>
    <div>
      <button @click="increase">Count is: {{ count }}</button>
    </div>
    <ur>
      <li>My cont is : {{ count }}</li>
    </ur>
    <hr />
    <prism-editor
      class="my-editor height-300"
      v-model="code"
      :highlight="highlighter"
      :line-numbers="lineNumbers"
      :readonly="readonlyType"
    >
    </prism-editor>
    <el-form>
      <input type="text" name="user" placeholder="用户名" />
      <input type="password" name="pwd" placeholder="密码" />
      <input type="submit" value="提交" />
    </el-form>

  </div>
</template>

<script>
import { PrismEditor } from "vue-prism-editor";
import "vue-prism-editor/dist/prismeditor.min.css"; // import the styles somewhere
// import highlighting library (you can use any library you want just return html string)
import { highlight, languages } from "prismjs/components/prism-core";
import "prismjs/components/prism-clike";
import "prismjs/components/prism-javascript";
import "prismjs/themes/prism-tomorrow.css"; // import syntax highlighting styles
export default {
  data() {
    return {
      count: 0,
      my_count: 0,
      n1: "西北工业大学",
      n2: ["西北工业大学", "计算机学院", "王羽展"],
      code: 'console.log("Hello World")\n' + 'console.log("Hello World")',
      lineNumbers: true, // true 显示行号   false 不显示行号
      readonlyType: true, //true不可编辑   false 可编辑
    };
  },
  components: {
    PrismEditor,
  },
  methods: {
    increase() {
      this.count++;
    },
    highlighter(code) {
      return highlight(code, languages.js); //returns html
    },
  },
};
</script>

<style scoped>
button {
  font-weight: bold;
}
.my-editor {
  background: #2d2d2d;
  color: #ccc;
  font-family: Fira code, Fira Mono, Consolas, Menlo, Courier, monospace;
  font-size: 14px;
  line-height: 1.5;
  padding: 5px;
}
.prism-editor__textarea:focus {
  outline: none;
}
</style>

