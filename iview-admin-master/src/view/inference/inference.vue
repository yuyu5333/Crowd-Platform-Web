<template>
    <div>
        <h2>处理视频流</h2>
        <Row :gutter="32">
            <i-col span="16">
                <Card>
                    <Upload multiple type="drag" action="//jsonplaceholder.typicode.com/posts/">
                        <div style="padding: 20px 0">
                            <Icon type="ios-cloud-upload" size="300" style="color: #3399ff"></Icon>
                            <p>点击或拖拽上传</p>
                        </div>
                    </Upload>
                </Card>
            </i-col>
            <i-col span="8">
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
            </i-col>
        </Row>


        <h2>推理结果</h2>
        <Table :height="500" :width="710" :align="center" />
    </div>
</template>

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