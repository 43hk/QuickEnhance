# QuickEnhance

## 🌟 项目简介

**QuickEnhance**是一个轻量级、开源的图像美化工具，基于QT和OpenCV。课程作业，供自己存档和与大家交流学习。

## 🚀 功能特性

- 基础：图像的亮度、对比度调整
- 平滑：提供四种算法：均值、高斯、中值、双边

## 📦 安装与运行

- 源码使用：
  - 下载源码，用QT Creater打开pro文件，配置项目
  - 点击构建，出现build文件夹
  - 将自己编译的opencv/install/x64/mingw/bin下的所有文件放入debug目录
  - 运行项目
- 软件使用：
  - 下载release文件，解压
  - 点击exe运行
  - 点击菜单的“文件”，进行图片的读取
  - 在右侧进行数值调整
  - 点击菜单的“文件”，进行图片的另存为或读取其他图片

## 📌 版本历史

| 版本   | 日期        | 说明                                   |
| ---- | --------- | ------------------------------------ |
| v1.0 | 2025.3.12 | 完成基础框架和基本功能，可以进行图像的亮度、对比度调整，以及四种平滑方式 |

### 📅 未来计划

- [ ] 修复平滑后切换基础功能调整时平滑消失的bug
- [ ] 图像锐化功能
- [ ] 直方图的显示和均衡化
- [ ] HSV调整
- [ ] 优化界面，使的控件可以缩放
- [ ] ······

### 📝 特别致谢

- QT教育许可
- OpenCV
