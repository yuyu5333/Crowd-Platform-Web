
<template>
    <div>
        <pre><h2><b>已启用设备</b>                                                                                                   <b>剩余可用设备</b></h2></pre>
        <Row :gutter="32">
            <i-col span="4">
                <Card>
                    <!-- <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/"> -->
                        <div style="padding: 20px 0; text-align:center">
                            <!-- <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon> -->
                            <p><b>ID:XXX</b></p>
                            <p style="color:dodgerblue">详细资源</p>
                        </div>
                    <!-- </Upload> -->
                </Card>
            </i-col>

            <i-col span="4">
                <Card>
                    <!-- <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/"> -->
                        <div style="padding: 20px 0; text-align:center">
                            <!-- <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon> -->
                            <p><b>ID:XXX</b></p>
                            <p style="color:dodgerblue">详细资源</p>
                        </div>
                    <!-- </Upload> -->
                </Card>
            </i-col>

            <i-col span="4">
                <Card>
                    <!-- <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/"> -->
                        <div style="padding: 20px 0; text-align:center">
                            <!-- <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon> -->
                            <p><b>ID:XXX</b></p>
                            <p style="color:dodgerblue">详细资源</p>
                        </div>
                    <!-- </Upload> -->
                </Card>
            </i-col>

            <i-col span="4">
                <Card style="color:dimgray">
                    <!-- <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/"> -->
                        <div style="padding: 20px 0; text-align:center">
                            <!-- <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon> -->
                            <p><b>ID:XXX</b></p>
                            <p style="color:dodgerblue">详细资源</p>
                        </div>
                    <!-- </Upload> -->
                </Card>
            </i-col>
            
            <i-col span="4">
                <Card style="color:dimgray">
                    <!-- <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/"> -->
                        <div style="padding: 20px 0; text-align:center">
                            <!-- <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon> -->
                            <p><b>ID:XXX</b></p>
                            <p style="color:dodgerblue">详细资源</p>
                        </div>
                    <!-- </Upload> -->
                </Card>
            </i-col>

            <i-col span="4">
                <Card style="color:dgray">
                    <!-- <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/"> -->
                        <div style="padding: 20px 0; text-align:center">
                            <!-- <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon> -->
                            <p><b>ID:XXX</b></p>
                            <p style="color:dodgerblue">详细资源</p>
                        </div>
                    <!-- </Upload> -->
                </Card>
            </i-col>
          
        </Row>
        <br>
        <div>
            <h2><b>深度模型选择</b></h2>                                                                                           
            <i-col >
                <div>
                    <card shadow title="模型选择" size="large"> 
                        <Form model="formItem" :label-width="80">
                            <FormItem label="任务分类">
                                <RadioGroup v-model="formItem.任务分类" size="large">
                                    <Radio label="任务1">图像分类</Radio>
                                    <Radio label="任务2">目标检测</Radio>
                                    <Radio label="任务3">其他任务</Radio>
                                </RadioGroup>
                            </FormItem>
                            <FormItem label="系统模型">
                                <RadioGroup v-model="formItem.系统模型" size="large">
                                    <Radio label="模型1">模型1</Radio>
                                    <Radio label="模型2">模型2</Radio>
                                    <Radio label="模型3">模型3</Radio>
                                </RadioGroup>
                            </FormItem>
                            <FormItem label="自定义模型">
                                <Upload action="//jsonplaceholder.typicode.com/posts/">
                                    <Button icon="ios-cloud-upload-outline" style="margin-left: 16px">上传文件</Button>
                                </Upload>
                            </FormItem>
                            <FormItem>
                                <Button type="primary" style="margin-left: 8px">感知模型</Button>
                            </FormItem>
                        </Form>
                    </card>
                </div>
            </i-col>
        </div>
        <br>
        <div>  
        <h2>推理结果</h2>
        <Table :height="300" :align="center" />
        </div>
    </div>
</template>

<Card>
    <Row>
        <Form model="formItem" :label-width="100">
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
        </Form>
    </Row>
    <Row>
        <Button type="primary" icon="ios-cloud-done-outline" shape="circle">开始推理</Button>
    </Row>

</Card>

<script>
export default {
    data() {
        return {
            formItem: {
                任务类型: "",
                模型选择: ""
            }
        }
    }

    
}
</script>