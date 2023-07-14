import Main from '@/components/main'
import parentView from '@/components/parent-view'

/**
 * iview-admin中meta除了原生参数外可配置的参数:
 * meta: {
 *  title: { String|Number|Function }
 *         显示在侧边栏、面包屑和标签栏的文字
 *         使用'{{ 多语言字段 }}'形式结合多语言使用，例子看多语言的路由配置;
 *         可以传入一个回调函数，参数是当前路由对象，例子看动态路由和带参路由
 *  hideInBread: (false) 设为true后此级路由将不会出现在面包屑中，示例看QQ群路由配置
 *  hideInMenu: (false) 设为true后在左侧菜单不会显示该页面选项
 *  notCache: (false) 设为true后页面在切换标签后不会缓存，如果需要缓存，无需设置这个字段，而且需要设置页面组件name属性和路由配置的name一致
 *  access: (null) 可访问该页面的权限数组，当前路由设置的权限会影响子路由
 *  icon: (-) 该页面在左侧菜单、面包屑和标签导航处显示的图标，如果是自定义图标，需要在图标名称前加下划线'_'
 *  beforeCloseName: (-) 设置该字段，则在关闭当前tab页时会去'@/router/before-close.js'里寻找该字段名对应的方法，作为关闭前的钩子函数
 * }
 */

export default [
  {
    path: '/login',
    name: 'login',
    meta: {
      title: 'Login - 登录',
      hideInMenu: true
    },
    component: () => import('@/view/login/login.vue')
  },
  {
    path: '/',
    name: '_home',
    redirect: '/home',
    component: Main,
    meta: {
      hideInMenu: true,
      notCache: true
    },
    children: [
      {
        path: '/home',
        name: 'home',
        meta: {
          hideInMenu: true,
          title: '首页',
          notCache: true,
          icon: 'md-home'
        },
        // component: () => import('@/view/single-page/home')
        component: () => import('@/view/single-page/home')
      }
    ]
  },
  /*  home page by wyz
  {
    path: '/hmthome',
    name: '_hmthome',
    meta: {
      hideInBread: true
    },
    component: Main,
    children: [
      {
        path: 'hmthome',
        name: 'hmthome',
        meta: {
          icon: 'ios-book',
          title: '首页'
        },
        component: () => import('@/view/hmthome/hmthome.vue')
      }
    ]
  },
  */
  {
    path: '/cog_natural',
    name: 'cog_natural',
    meta: {
      hideInBread: true
    },
    component: Main,
    children: [
      {
        path: 'cog_natural',
        name: 'cog_natural',
        meta: {
          icon: 'md-planet',
          title: '资源感知'
        },
        component: () => import('@/view/cognition/cog_natural.vue')
      }
    ]
  },
  {
    path: '/ModelPerformaceEvaluation',
    name: '_ModelPerformaceEvaluation',
    meta: {
      icon: 'md-cloud-upload',
      title: '模型性能评估'
    },
    component: Main,
    children: [
      {
        path: 'firstpage',
        name: 'firstpage',
        meta: {
          icon: 'md-planet',
          title: '功能说明'
        },
        component: () => import('@/view/ModelPerformanceEvaluation/firstpage.vue')
      },
      {
        path: 'Predefined_model',
        name: 'Predefined_model',
        meta: {
          icon: 'ios-document',
          title: '系统模型'
        },
        component: () => import('@/view/ModelPerformanceEvaluation/Predefined_model.vue')
      },
      // {
      //   path: 'User_defined_model',
      //   name: 'User_defined_model',
      //   meta: {
      //     icon: 'md-clipboard',
      //     title: '上传模型'
      //   },
      //   component: () => import('@/view/ModelPerformanceEvaluation/User_defined_model.vue')
      // },
      // {
      //   path: 'model-test1',
      //   name: 'model-test1',
      //   meta: {
      //     icon: 'md-clipboard',
      //     title: '测试1'
      //   },
      //   component: () => import('@/view/ModelPerformanceEvaluation/test.vue')
      // },
      // {
      //   path: 'model-test2',
      //   name: 'model-test2',
      //   meta: {
      //     icon: 'md-clipboard',
      //     title: '测试2'
      //   },
      //   component: () => import('@/view/ModelPerformanceEvaluation/test2.vue')
      // },
    ]
  },
  {
    path: '/ModelCompress',
    name: 'ModelCompress',
    component: Main,
    meta: {
      hideInBread: true
    },
    children: [
      {
        path: 'ModelCompress',
        name: 'ModelCompress',
        meta: {
          icon: 'md-grid',
          title: '模型压缩'
        },
        component: () => import('@/view/ModelCompress/SystemModel.vue')
      }
    ]
  }, 
  {
    path: '/segmentation',
    name: 'segmentation',
    component: Main,
    meta: {
      hideInBread: true
    },
    children: [
      {
        path: 'segmentation',
        name: 'segmentation',
        meta: {
          icon: 'ios-bug',
          title: '模型分割'
        },
        component: () => import('@/view/model_segmentation/segmentation_latency.vue')
      }
    ]
  }, 
]
