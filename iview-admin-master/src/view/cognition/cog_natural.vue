<template>
    <div id="app">
          <el-container>
            <!--<el-header>
              <div style="margin-top: 15px;">
                  <el-progress :percentage="percentage" :color="customColor"></el-progress>
              </div>
          </el-header>-->
            <el-container>
              <el-aside width="250px">
                <div style="margin-top: 10px;height: 65px;">
                    <div style="margin-left: 5px">
                        <div class="firstBlock"><el-avatar :size="50" :src="imgUrl"></el-avatar></div>
                        <div class="secondBlock">{{this.name}}</div>
                        <div class="thirdBlock"><el-button type="primary" circle icon="el-icon-video-play" @click="openEcharts"></el-button></div>
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
                            <td>CPU Architecture</td>
                            <td>{{this.CPU_Arch}}</td>
                        </tr>
                        <tr>
                            <td>CPU Type</td>
                            <td>{{this.CPU_Type}}</td>
                        </tr>
                        <tr>
                            <td>OS Version</td>
                            <td>{{this.OS_Version}}</td>
                        </tr>
                        <tr>
                            <td>Physical Memory</td>
                            <td>{{this.RAM_Total}}</td>
                        </tr>
                        <tr>
                            <td>GPU Type</td>
                            <td>{{this.GPU_Type}}</td>
                        </tr>
                    </table>
                </div>  
            </el-aside>
            
              <el-main>
                <div id="main" style="width: 800px;height:1500px;margin-left: 0px;">
                    <card style="width: 800px;height: 370px;margin-bottom: 10px;">
                        <h1 style="color:#2867a8">CPU Usage</h1>
                        <CPU_usage></CPU_usage>
                    </card>
                    
                    <card style="width: 800px;height: 370px;margin-bottom: 10px;">
                        <h1 style="color:#2867a8">GPU Usage</h1>
                        <GPU_usage></GPU_usage>
                    </card>

                    <card style="width: 800px;height: 370px;margin-bottom: 10px;">
                        <h1 style="color:#2867a8">Memory Usage</h1>
                        <Memory_usage></Memory_usage>
                    </card>

                    <card style="width: 800px;height: 370px;margin-bottom: 10px;">
                        <h1 style="color:#2867a8">Disk Free</h1>
                        <Disk_free></Disk_free>
                    </card>
                </div>
            </el-main>
            
            </el-container>
          </el-container>
    </div>
    </template>
    
    <script>
    import { inject} from "vue";
    import CPU_usage from "@/components/natural/CPU_usage.vue";
    import GPU_usage from "@/components/natural/GPU_usage.vue";
    import Memory_usage from "@/components/natural/Memory_usage.vue";
    import Disk_free from "@/components/natural/Disk_free.vue";
    import imgUrl from '@/assets/images/dog.jpg';
    import axios from "axios";
    export default {
        name:'cog_natural',
        components:{CPU_usage,GPU_usage,Memory_usage,Disk_free},
        data() {
              return {
                 DeviceOptions: [
                        {
                          value: '选项1',
                          label: 'Device1',
                        }, 
                        {
                          value: '选项2',
                          label: 'Device2'
                        }, 
                        {
                          value: '选项3',
                          label: 'Device3'
                        }, 
                        {
                          value: '选项4',
                          label: 'Device4'
                        }, 
                        {
                          value: '选项5',
                          label: 'Device5'
                        }],
                        value: '选择测试设备',
                        device:'',
                        CPU_Arch:'',
                        CPU_Type:'',
                        GPU_Type:'',
                        OS_Version:'',
                        RAM_Total:'',
                        imgUrl:imgUrl,
                        name:'李瑶',
                        //percentage: 20,
                        //customColor: '#409eff',
              }
          },
          mounted(){
          },
          watch:{},
          methods:{
            checkDeviceChange(device){
              let that = this;
              if (device == '选项1'){
                axios.get('/deviceinfo/').then(response => {
                that.CPU_Type = response.data.CPU_Type;
                that.GPU_Type = response.data.GPU_Type;
                that.CPU_Arch = response.data.CPU_Arch;
                that.OS_Version = response.data.OS_Version;
                that.RAM_Total = response.data.RAM_Total;
              })
              }
              else{
                that.CPU_Type = '';
                that.GPU_Type = '';
                that.CPU_Arch = '';
                that.OS_Version = '';
                that.RAM_Total = '';
              }
              
            },
            /*openEcharts(){
                // 基于准备好的dom，初始化echarts实例
                let $echarts = inject("echarts");
                var myChart = $echarts.init(document.getElementById('chart'));
                            
                // prettier-ignore
                const data = [["2000-06-05", 116], ["2000-06-06", 129], ["2000-06-07", 135], ["2000-06-08", 86], ["2000-06-09", 73], ["2000-06-10", 85], ["2000-06-11", 73], ["2000-06-12", 68], ["2000-06-13", 92], ["2000-06-14", 130], ["2000-06-15", 245], ["2000-06-16", 139], ["2000-06-17", 115], ["2000-06-18", 111], ["2000-06-19", 309], ["2000-06-20", 206], ["2000-06-21", 137], ["2000-06-22", 128], ["2000-06-23", 85], ["2000-06-24", 94], ["2000-06-25", 71], ["2000-06-26", 106], ["2000-06-27", 84], ["2000-06-28", 93], ["2000-06-29", 85], ["2000-06-30", 73], ["2000-07-01", 83], ["2000-07-02", 125], ["2000-07-03", 107], ["2000-07-04", 82], ["2000-07-05", 44], ["2000-07-06", 72], ["2000-07-07", 106], ["2000-07-08", 107], ["2000-07-09", 66], ["2000-07-10", 91], ["2000-07-11", 92], ["2000-07-12", 113], ["2000-07-13", 107], ["2000-07-14", 131], ["2000-07-15", 111], ["2000-07-16", 64], ["2000-07-17", 69], ["2000-07-18", 88], ["2000-07-19", 77], ["2000-07-20", 83], ["2000-07-21", 111], ["2000-07-22", 57], ["2000-07-23", 55], ["2000-07-24", 60]];
                const dateList = data.map(function (item) {
                  return item[0];
                });
                const valueList = data.map(function (item) {
                  return item[1];
                });
                option = {
                  // Make gradient line here
                  visualMap: [
                    {
                      show: false,
                      type: 'continuous',
                      seriesIndex: 0,
                      min: 0,
                      max: 400
                    },
                    {
                      show: false,
                      type: 'continuous',
                      seriesIndex: 1,
                      dimension: 0,
                      min: 0,
                      max: dateList.length - 1
                    }
                  ],
                  title: [
                    {
                      left: 'center',
                      text: 'Gradient along the y axis'
                    },
                    {
                      top: '55%',
                      left: 'center',
                      text: 'Gradient along the x axis'
                    }
                  ],
                  tooltip: {
                    trigger: 'axis'
                  },
                  xAxis: [
                    {
                      data: dateList
                    },
                    {
                      data: dateList,
                      gridIndex: 1
                    }
                  ],
                  yAxis: [
                    {},
                    {
                      gridIndex: 1
                    }
                  ],
                  grid: [
                    {
                      bottom: '60%'
                    },
                    {
                      top: '60%'
                    }
                  ],
                  series: [
                    {
                      type: 'line',
                      showSymbol: false,
                      data: valueList
                    },
                    {
                      type: 'line',
                      showSymbol: false,
                      data: valueList,
                      xAxisIndex: 1,
                      yAxisIndex: 1
                    }
                  ]
                };
                            
                // 使用刚指定的配置项和数据显示图表。
                myChart.setOption(option,true);
            },*/
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
                font-size: large;
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