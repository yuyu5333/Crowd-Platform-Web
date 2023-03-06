<template>
    <div class="image-container">
        <div class="images">
            <img v-for="item in images" :key="item.index" :src="item.path" :class="{active: item.active}">
        </div>
  </div>
</template>

<script>
import Raspberry from '@/assets/images/raspberry.jpg';
import Jetson from '@/assets/images/jetson.jpg';
import Android from '@/assets/images/android.jpg';
import Mcu from '@/assets/images/mcu.jpg';
import Fpga from '@/assets/images/fpga.png';
export default {
    name: 'show_img',
    data(){
        return {
            images:[
                {index:1,path:Raspberry,name:'Raspberry',active:false},
                {index:2,path:Jetson,name:'Jetson',active:false},
                {index:3,path:Android,name:'Android',active:false},
                {index:4,path:Mcu,name:'Mcu',active:false},
                {index:5,path:Fpga,name:'Fpga',active:false}
            ],
        }
    },
    methods:{
        getChange(device){
            for(var i = 0;i < this.images.length;i++){
                var item = this.images[i];
                console.log(device);
                if(item.name == device){
                    item.active = true;
                }
                else{
                    item.active = false;
                }
            }
            this.$forceUpdate();
        }
    }
}
</script>

<style>
.image-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
}

.images {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: calc(100% - 10px);
  margin-bottom: 10px;
}

img {
  width: calc(20% - 5px);
  height: 200px;
  filter: grayscale(100%);
}
.active{
    border:1px solid rgb(152, 191, 250);
    box-shadow:0 0 20px rgb(152, 191, 250);
    filter: grayscale(0%);
    animation: jump 0.8s;
}
@keyframes jump {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
  100% {
    transform: scale(1);
  }
}
/*@keyframes jump {
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
  100% {
    transform: translateY(0);
  }
}*/

</style>