<template>
    <div id="myChart_gpu"></div>
</template>

<script>
import echarts from 'echarts';
import axios from "axios";
export default {
    name: 'GPU_usage',
    data () {
        return {
        	// 实时数据数组
            date: [],
            GPU_Use: [],
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
                    name: 'GPU使用率',
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
                        formatter: '{value} %'
                    }
                },
                tooltip: {
                    trigger: 'axis',
                },
                series: [
                    {
                        name:'GPU使用率',
                        type:'line',
                        smooth: true,
                        data: this.GPU_Use,	// 绑定实时数据数组
                    },
                ]
            }
        }
    },
    mounted () {
        this.myChart = echarts.init(document.getElementById('myChart_gpu'), 'light');	// 初始化echarts, theme为light
        this.myChart.setOption(this.echartsOption);	// echarts设置初始化选项
        setInterval(this.addData, 3000);	// 每三秒更新实时数据到折线图
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
        // 添加实时数据
        addData : function() {
            //let that = this;
            axios.get('/resourceinfo/').then(response => {
                this.GPU_Use.push(parseFloat(response.data.GPU_Use).toFixed(3));
                this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                this.echartsOption.xAxis.data = this.date;
                this.echartsOption.series[0].data = this.GPU_Use;
                this.myChart.setOption(this.echartsOption);
            });
        	// 从接口获取数据并添加到数组
            /*this.$axios.get('url').then((res) => {
                this.yieldRate.push((res.data.actualProfitRate * 100).toFixed(3));
                this.yieldIndex.push((res.data.benchmarkProfitRate * 100).toFixed(3));
                this.date.push(this.getTime(Math.round(new Date().getTime() / 1000)));
                // 重新将数组赋值给echarts选项
                this.echartsOption.xAxis.data = this.date;
                this.echartsOption.series[0].data = this.yieldRate;
                this.echartsOption.series[1].data = this.yieldIndex;
                this.myChart.setOption(this.echartsOption);
            });*/
        }
    }
}
</script>

<style>
#myChart_gpu{
  width: 800px;
  height: 330px;
  margin: 0 auto;
}
</style>
