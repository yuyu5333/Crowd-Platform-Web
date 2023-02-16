<template>
    <div class="demo-split">
        <Split v-model="split1">
            <template #left>
                <div class="demo-split-pane no-padding">
                    <Split v-model="split2" mode="vertical">
                        <template #top>
                            <div class="demo-split-pane">
                                <card shadow title="已启用设备" size="large">
                                    <Row>
                                        <Col span="8">
                                            <p>ID:XXX</p>
                                        </Col>
                                        <Col span="8">
                                            <p>ID:XXX</p>
                                        </Col>
                                        <Col span="8">
                                            <p>ID:XXX</p>
                                        </Col> 
                                    </Row>
                                    <br></br>
                                    <br></br>
                                                                       
                                </card>
                            </div>
                        </template>
                        <template #bottom>
                            <div class="demo-split-pane">
                                <card shadow title="剩余可用设备" size="large">
                                    <Row>
                                        <Col span="8">
                                            <p>ID:XXX</p>
                                        </Col>
                                        <Col span="8">
                                            <p>ID:XXX</p>
                                        </Col>
                                        <Col span="8">
                                            <p>ID:XXX</p>
                                        </Col> 
                                    </Row>
                                    <br></br>
                                    <br></br>
                                </card>
                            </div>
                        </template>
                    </Split>
                </div>
            </template>
           <template #right>
                <div class="demo-split-pane">
                    <card shadow title="模型选择" size="large"> 
                        <Form model="formItem" :label-width="80">
                            <FormItem label="系统模型">
                                <RadioGroup v-model="formItem.系统模型" size="large">
                                    <Radio label="模型1">模型1</Radio>
                                    <Radio label="模型2">模型2</Radio>
                                    <Radio label="模型3">模型3</Radio>
                                </RadioGroup>
                            </FormItem>
                            <FormItem label="自定义模型">
                                <Upload action="//jsonplaceholder.typicode.com/posts/">
                                    <Button icon="ios-cloud-upload-outline" style="margin-left: 16px">Upload files</Button>
                                </Upload>
                            </FormItem>
                            <FormItem>
                                <Button type="primary" style="margin-left: 8px">感知模型</Button>
                            </FormItem>
                        </Form>
                    </card>
                </div>
            </template>
        </Split>
    </div>
</template>

<script>
    export default {
        data () {
            return {
                split1: 0.5,
                split2: 0.5,
                formItem:{
                    系统模型:'',

                }

            }
        },
    }
</script>
<style>
    .demo-split{
        height: 500px;
        border: 1px solid #dcdee2;
    }
    .demo-split-pane{
        padding: 10px;
    }
    .demo-split-pane.no-padding{
        height: 300px;
        padding: 0;
    }
</style>