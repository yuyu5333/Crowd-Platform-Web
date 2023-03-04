<template>
    <div id="app">
            <el-container>
              <el-aside width="250px">
                <div style="margin-top: 10px;height: 65px;">
                    <div style="margin-left: 5px">
                        <div class="firstBlock"><el-avatar :size="50" :src="imgUrl"></el-avatar></div>
                        <div class="secondBlock">{{this.name}}</div>
                        <div class="thirdBlock"><el-button type="primary" circle icon="el-icon-video-play"></el-button></div>
                    </div>
                </div>
                <hr>
                
                <Select v-model="device" @on-change="checkDeviceChange" placeholder="请选择测试设备" size="large" style="width:250px;margin-top: 10px;">
                  <Option v-for="item in DeviceOptions" :value="item.value" :key="item.value">{{ item.label }}</Option>
                </Select>
                
                
                <div style="margin-top: 10px;">
                    <table>
                        <tr class="first">
                            <td>Info</td>
                            <td>Value</td>
                        </tr>
                        <tr>
                            <td style="font-size:larger;">CPU Architecture</td>
                            <td>{{this.CPU_Arch}}</td>
                        </tr>
                        <tr>
                            <td style="font-size:larger;">CPU Type</td>
                            <td>{{this.CPU_Type}}</td>
                        </tr>
                        <tr>
                            <td style="font-size:larger;">OS Version</td>
                            <td>{{this.OS_Version}}</td>
                        </tr>
                        <tr>
                            <td style="font-size:larger;">Physical Memory</td>
                            <td>{{this.RAM_Total}}</td>
                        </tr>
                        <tr>
                            <td style="font-size:larger;">GPU Type</td>
                            <td>{{this.GPU_Type}}</td>
                        </tr>
                    </table>
                </div>  
            </el-aside>
            <el-main>
                <div id="main" style="height:1500px;margin-left: 0px;">
                    <card style="height: 300px;margin-bottom: 10px;">
                      <Row>
                        <h1 style="color:#2867a8">Show pictures</h1>
                      </Row>
                      <Row>
                        <Col :md="24" :lg="24" :xl="24">
                          <show_img ref="showImage"></show_img>
                        </Col>
                      </Row>
                    </card>
                    <card style="height: 370px;margin-bottom: 10px;">
                      <Row>
                        <h1 style="color:#2867a8">CPU Usage</h1>
                      </Row>
                      <Row>
                        <Col :md="24" :lg="24" :xl="24">
                          <CPU_usage :DeviceName=device style="height:330px"></CPU_usage>
                        </Col>
                      </Row>  
                    </card>
                    
                    <card style="height: 370px;margin-bottom: 10px;">
                      <Row>
                        <h1 style="color:#2867a8">GPU Usage</h1>
                      </Row>
                      <Row>
                        <Col :md="24" :lg="24" :xl="24">
                          <GPU_usage :DeviceName=device style="height:330px"></GPU_usage>
                        </Col>
                      </Row>
                        
                    </card>

                    <card style="height: 370px;margin-bottom: 10px;">
                      <Row>
                        <h1 style="color:#2867a8">Memory Usage</h1>
                      </Row>
                      <Row>
                        <Col :md="24" :lg="24" :xl="24">
                          <Memory_usage :DeviceName=device style="height:330px"></Memory_usage>
                        </Col>
                      </Row>
                    </card>

                    <card style="height: 370px;margin-bottom: 10px;">
                      <Row>
                        <h1 style="color:#2867a8">Disk Free</h1>
                      </Row>
                      <Row>
                        <Col :md="24" :lg="24" :xl="24">
                          <Disk_free :DeviceName=device style="height:330px"></Disk_free>
                        </Col>
                      </Row>
                    </card>
                </div>
            </el-main>
            </el-container>
          
    </div>
    </template>
    
    <script>
    import { inject} from "vue";
    import CPU_usage from "@/components/natural/CPU_usage.vue";
    import GPU_usage from "@/components/natural/GPU_usage.vue";
    import Memory_usage from "@/components/natural/Memory_usage.vue";
    import Disk_free from "@/components/natural/Disk_free.vue";
    import show_img from "@/components/natural/show_img.vue";
    import imgUrl from '@/assets/images/dog.jpg';
    import axios from "axios";
    export default {
        name:'cog_natural',
        components:{CPU_usage,GPU_usage,Memory_usage,Disk_free,show_img},
        data() {
              return {
                DeviceOptions: [
                  {
                    value: 'test',
                    label: 'test',
                  }, 
                  {
                    value: 'Raspberry',
                    label: 'Raspberry'
                  }, 
                  {
                    value: 'Jetson',
                    label: 'Jetson'
                  }, 
                  {
                    value: 'Device4',
                    label: 'Device4'
                  }, 
                  {
                    value: 'Device5',
                    label: 'Device5'
                  }
                ],
                value: '选择测试设备',
                device:'',
                CPU_Arch:'',
                CPU_Type:'',
                GPU_Type:'',
                OS_Version:'',
                RAM_Total:'',
                imgUrl:imgUrl,
                name:'李瑶',
              }
        },
        mounted(){
        },
        watch:{},
        methods:{
          checkDeviceChange(device){
            let that = this;
            //const img_show = that.$refs.img_show;
            //img_show.getDeviceName();
            //this.$refs.show_img.getDeviceName();
            //that.$refs.show_img.getChange(device);
            that.$refs.showImage.getChange(device);
            axios.post('/deviceinfo/',{
              DeviceName:device
            }).then(response => {
              that.CPU_Type = response.data.CPU_Type;
              that.GPU_Type = response.data.GPU_Type;
              that.CPU_Arch = response.data.CPU_Arch;
              that.OS_Version = response.data.OS_Version;
              that.RAM_Total = response.data.RAM_Total;
            })
            
          },
        },
    
    }
    </script>
    
    <style scoped>
    .el-aside {
              background-color: #D3DCE6;
              color: #333;
              text-align: center;
            }
            
            .el-main {
              background-color: #E9EEF3;
              color: #333;
              text-align: center;
            }
            
            body > .el-container {
              margin-bottom: 20px;
            }
            .el-table__body-wrapper::-webkit-scrollbar {
                display: none;
            }
            table {
                width: 100%;
                background: #ccc;
                margin: 10px auto;
                border-collapse: collapse;
                font-family: "黑体";
            }
            th,
            td {
                height: 25px;
                line-height: 25px;
                text-align: center;
                border: 1px solid #ccc;
            }
            th {
                background: #eee;
                font-weight: normal;
            }
            tr {
                background: #fff;
            }
            td a {
                color: #06f;
                text-decoration: none;
            }
            td a:hover {
                color: #06f;
                text-decoration: underline;
            }
            
            .first{
                font-size: x-large;
                color: darkgrey;
            }
            
            .secondBlock{
                font-family: "verdana, geneva, arial, helvetica, sans-serif";
                color: darkslateblue;
                text-align: left;
                margin-left: 55px;
            }
            
            .firstBlock,.SecondBlock{
                float: left;
            }
            
            .thirdBlock{
                float: right;
                margin-bottom: 10px;
            }
    </style>