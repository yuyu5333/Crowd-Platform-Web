<template>
  <div class="image-container">
    <img id="pic" v-for="item in images" :key="item.index" v-if="item.active==true" :src="item.path">
  </div>
</template>

<script>
import Raspberry from '@/assets/images/rasp3.jpg';
import Jetson from '@/assets/images/jetson2.jpg';
import Android from '@/assets/images/jiqi1.jpg';
import Mcu from '@/assets/images/mcu4.jpg';
import Fpga from '@/assets/images/fpga.png';
export default {
    name: 'showpicture',
    data(){
        return{
            images:[
            {index:1,path:Raspberry,name:'Raspberry',active:false},
            {index:2,path:Jetson,name:'Jetson',active:false},
            {index:3,path:Android,name:'Android',active:false},
            {index:4,path:Mcu,name:'Mcu',active:false},
            {index:5,path:Fpga,name:'Fpga',active:false}
            ]
        }
    },
    methods:{
        getChange(device){
            for(var i = 0;i < this.images.length;i++){
                var item = this.images[i];
                //console.log(device);
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
#pic {
    width:250px;
    height: 250px;
    filter: grayscale(0%);
    border-radius: 50%;
}

</style>