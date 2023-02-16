
<template>
    <div>
        <div>
            <card shadow title="选择部署任务" size="large">
            <Row>
            <Form model="formItem" :label-width="80">
            <FormItem label="任务类型">
                <RadioGroup v-model="formItem.任务类型" size="large">
                    <Radio label="图像分类">图像分类</Radio>
                    <Radio label="目标检测">目标检测</Radio>
                    <Radio label="其他任务">其他任务</Radio>
                </RadioGroup>
            </FormItem>
            <FormItem label="模型选择">
                <RadioGroup v-model="formItem.模型选择" size="large">
                    <Radio label="模型1">模型1</Radio>
                    <Radio label="模型2">模型2</Radio>
                    <Radio label="模型3">模型3</Radio>
                </RadioGroup>
            </FormItem>
            <FormItem label="压缩方法">
                <RadioGroup v-model="formItem.压缩方法" size="large">
                    <Radio label="剪枝">剪枝</Radio>
                    <Radio label="量化">量化</Radio>
                    <Radio label="知识蒸馏">知识蒸馏</Radio>
                </RadioGroup>
            </FormItem>        
            <FormItem label="压缩率">
                <Slider v-model="formItem.slider" range></Slider>
            </FormItem>
            <FormItem>
                <Button type="primary">Submit</Button>
                <Button style="margin-left: 8px">Cancel</Button>
            </FormItem>
            </Form>
        </Row>
        </card>
        </div>
 <div>
    <br>
 </div>   

<div>
    <card shadow title="效果展示" size="large">
        <Col span="8">
            
        </Col>
        <Col span="8">

        </Col>
        <Col span="8">

        </Col>    
    </card>
</div>
        
    </div>
</template>
<script>
    export default {

        data () {
            return {
                formItem: {
                    任务类型: '',
                    模型选择:'',
                    压缩方法:'',
                    压缩率: [0, 0],

                }
            }
        }
    }
</script>

