<style lang="less">
@import './login.less';
</style>

<template>
  <div class="login">
    <div class="login-con">
      <div class="ivu-card">
        <div class="ivu-card-head">
          <p><i class="ivu-icon ivu-icon-log-in"></i> <span>欢迎登录</span></p>
        </div>
        <!---->
        <div class="ivu-card-body">
          <div class="form-con">
            <form autocomplete="off" class="ivu-form ivu-form-label-right">
              <div class="ivu-form-item ivu-form-item-required">
                <!---->
                <div class="ivu-form-item-content">
                  <div
                    class="ivu-input-wrapper ivu-input-wrapper-default ivu-input-type ivu-input-group ivu-input-group-default ivu-input-group-with-prepend">
                    <div class="ivu-input-group-prepend" style=""><span><i class="ivu-icon ivu-icon-ios-person"
                          style="font-size: 16px;"></i></span></div>
                    <div class="ivu-input-inner-container">
                      <!----> <i
                        class="ivu-icon ivu-icon-ios-loading ivu-load-loop ivu-input-icon ivu-input-icon-validate"></i>
                      <input autocomplete="off" spellcheck="false" type="text" placeholder="请输入用户名"
                        class="ivu-input ivu-input-default">
                    </div>
                    <!---->
                  </div>
                  <!---->
                </div>
              </div>
              <div class="ivu-form-item ivu-form-item-required">
                <!---->
                <div class="ivu-form-item-content">
                  <div
                    class="ivu-input-wrapper ivu-input-wrapper-default ivu-input-type ivu-input-group ivu-input-group-default ivu-input-group-with-prepend">
                    <div class="ivu-input-group-prepend" style=""><span><i class="ivu-icon ivu-icon-md-lock"
                          style="font-size: 14px;"></i></span>
                    </div>
                    <div class="ivu-input-inner-container">
                      <!----> <i
                        class="ivu-icon ivu-icon-ios-loading ivu-load-loop ivu-input-icon ivu-input-icon-validate"></i>
                      <input autocomplete="off" spellcheck="false" type="password" placeholder="请输入密码"
                        class="ivu-input ivu-input-default">
                    </div>
                    <!---->
                  </div>
                  <!---->
                </div>
              </div>
              <div class="ivu-form-item">
                <!---->
                <div class="ivu-form-item-content">
                    <button type="button" class="ivu-btn ivu-btn-primary ivu-btn-long">
                    <!----> <!----> <span>登录</span>
                    </button>
                  <!---->
                </div>
              </div>
            </form>
            <p class="login-tip">输入任意用户名和密码即可</p>
          </div>
        </div>
      </div>
    </div>
    <!-- built files will be auto injected -->
    <div class="ivu-loading-bar" style="height: 2px; display: none;">
      <div class="ivu-loading-bar-inner ivu-loading-bar-inner-color-primary" style="width: 0%; height: 2px;"></div>
    </div>
  </div>
</template>

<script>
import LoginForm from '_c/login-form'
import { mapActions } from 'vuex'
export default {
  components: {
    LoginForm
  },
  methods: {
    ...mapActions([
      'handleLogin',
      'getUserInfo'
    ]),
    handleSubmit({ userName, password }) {
      this.handleLogin({ userName, password }).then(res => {
        this.getUserInfo().then(res => {
          this.$router.push({
            name: this.$config.homeName
          })
        })
      })
    },
    gohome() {
      this.$router.push('/home')
    }
  }
}
</script>

<style></style>

