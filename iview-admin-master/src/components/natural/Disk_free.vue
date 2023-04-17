<template>
    <div id="myChart_disk"></div>
</template>

<script>
import echarts from 'echarts';
import axios from "axios";
export default {
    name: 'Disk_free',
    props:{
        DeviceName: String,
        required: true
    },
    data () {
        return {
        	// 实时数据数组
            date: [],
            DISK_Free: [],
            // 折线图echarts初始化选项
            echartsOption: {
                xAxis: {
                    name: '时间',
                    nameTextStyle: {
                        fontWeight: 600,
                        fontSize: 14
                    },
                    type: 'category',
                    boundaryGap: false,
                    data: this.date,	// 绑定实时数据数组
                },
                yAxis: {
                    name: '磁盘剩余空间',
                    nameTextStyle: {
                        fontWeight: 600,
                        fontSize: 14
                    },
                    type: 'value',
                    scale: true,
                    //boundaryGap: ['15%', '15%'],
                    boundaryGap:false,
                    axisLabel: {
                        interval: 'auto',
                        formatter: '{value} G'
                    }
                },
                tooltip: {
                    trigger: 'axis',
                },
                series: [
                    {
                        name:'磁盘剩余空间',
                        type:'line',
                        smooth: true,
                        data: this.DISK_Free,	// 绑定实时数据数组
                    },
                ]
            }
        }
    },
    mounted () {
        this.myChart = echarts.init(document.getElementById('myChart_disk'), 'light');	// 初始化echarts, theme为light
        this.myChart.setOption(this.echartsOption);	// echarts设置初始化选项
        setTimeout(()=>{
            this.myChart.resize();
        },100)
        setInterval(this.addData, 5000);	// 每1秒更新实时数据到折线图
    },
    methods: {
    	// 获取当前时间
        getTime : function() {	
            var ts = arguments[0] || 0;
            var t, h, i, s;
            t = ts ? new Date(ts * 1000) : new Date();
            h = t.getHours();
            i = t.getMinutes();
            s = t.getSeconds();
            // 定义时间格式
            return (h < 10 ? '0' + h : h) + ':' + (i < 10 ? '0' + i : i) + ':' + (s < 10 ? '0' + s : s);
        },
        getChange : function() {
            this.DISK_Free=[];
            this.date=[];
            this.echartsOption.xAxis.data = this.date;
            this.echartsOption.series[0].data = this.DISK_Free;
            this.myChart.setOption(this.echartsOption);
        },
        // 添加实时数据
        addData : function() {
            /*axios.post('/resourceinfo/',{
                DeviceName:this.DeviceName
            }).then(response => {
                this.DISK_Free.push(parseFloat(response.data.DISK_Free));
                this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                this.echartsOption.xAxis.data = this.date;
                this.echartsOption.series[0].data = this.DISK_Free;
                this.myChart.setOption(this.echartsOption);
            });*/
            if(this.DeviceName == 'Raspberry'){
                axios.get('/raspberry/').then(response => {
                    this.DISK_Free.push(parseFloat(response.data.DISK_Free));
                    this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                    this.echartsOption.xAxis.data = this.date;
                    this.echartsOption.series[0].data = this.DISK_Free;
                    this.myChart.setOption(this.echartsOption);
                })
            }
            else if(this.DeviceName == 'Jetson'){
                axios.get('/jetson/').then(response => {
                    this.DISK_Free.push(parseFloat(response.data.DISK_Free));
                    this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                    this.echartsOption.xAxis.data = this.date;
                    this.echartsOption.series[0].data = this.DISK_Free;
                    this.myChart.setOption(this.echartsOption);
                })
            }
            else if(this.DeviceName == 'Android'){
                axios.get('/android/').then(response => {
                    this.DISK_Free.push(parseFloat(response.data.DISK_Free));
                    this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                    this.echartsOption.xAxis.data = this.date;
                    this.echartsOption.series[0].data = this.DISK_Free;
                    this.myChart.setOption(this.echartsOption);
                })
            }
            else{
                this.DISK_Free.push(0);
                this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                this.echartsOption.xAxis.data = this.date;
                this.echartsOption.series[0].data = this.DISK_Free;
                this.myChart.setOption(this.echartsOption);
            }
        }
    }
}
</script>

<style>

</style>
