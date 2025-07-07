# <img src="https://s21.ax1x.com/2025/07/01/pVu3xA0.md.jpg" width="30">WUST_VISION



## 依赖
* OpenCV
* [OpenVINO](https://flowus.cn/7a2a3341-74a1-4db9-bced-99fe5d05ab75)/[TensorRT-cuda](https://flowus.cn/e98af178-de0b-4546-808d-a6f1ff199d62)/[NCNN](https://flowus.cn/664f6bee-8ea9-4d54-8a78-e2c0bf38ee9f)连接为简单部署文档
* fmt
* ceres
* Eigen3
* Sophus
* g2o
* nlohmann
* yaml-cpp
## 环境配置
```
sudo apt install libfmt-dev
sudo apt install libceres-dev
sudo apt install libeigen3-dev
sudo apt install nlohmann-json3-dev
sudo apt install libyaml-cpp-dev
```
Sophus: 
```
git clone https://github.com/strasdat/Sophus
cd Sophus
mkdir build && cd build
cmake ..
make -j #可以改cmakelists把test的编译关了，make就没必要了
sudo make install
```
G2O:
```
sudo apt install libeigen3-dev libspdlog-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5
git clone https://github.com/RainerKuemmerle/g2o
cd g2o
mkdir build && cd build
cmake ..
make -j
sudo make install
```
## Quick Start
```
sudo ./run.sh trt/openvino/ncnn/build #TensorRT-cuda识别版本/OpenVINO识别版本/NCNN识别版本/仅编译
```
### 注意：本项目默认要求OpenVINO、TensorRT-cuda或NCNN参与编译（可选择其一，需在build第一次缓存前在cmakelists设置）OpenVINO、TensorRT-cuda或NCNN实际参与装甲板与能量机关的识别，装甲板识别可使用纯OpenCV（但OpenVINO、TensorRT-cuda或NCNN仍然在编译路径中），能量机关如不使用可删除，OpenVINO、TensorRT-cuda版本可选择使用NCNN的装甲板/能量机关识别。
## 文件树
```
.
├── CMakeLists.txt
├── config # 配置文件
│   ├── armor_detect_opencv.yaml 
│   ├── camera_info.yaml
│   ├── config_ncnn.yaml
│   ├── config_openvino.yaml
│   ├── config_trt.yaml
│   ├── detect_ncnn.yaml
│   ├── guard_ncnn.sh
│   ├── guard_openvino.sh
│   └── guard_trt.sh
├── dandao.py
├── format.sh
├── include
│   ├── common # 通用
│   │   ├── 3rdparty # 第三方库
│   │   │   ├── angles.h
│   │   │   └── matplotlibcpp.h
│   │   ├── calculation.hpp
│   │   ├── debug # 调试
│   │   │   ├── matplottools.hpp
│   │   │   ├── toolsgobal.hpp
│   │   │   └── tools.hpp
│   │   ├── gobal.hpp
│   │   ├── logger.hpp
│   │   ├── tf.hpp
│   │   ├── ThreadPool.h
│   │   └── utils.hpp
│   ├── control # 控制
│   │   ├── armor_solver.hpp
│   │   ├── control_filter.hpp
│   │   ├── manual_compensator.hpp
│   │   ├── rune_solver.hpp
│   │   └── trajectory_compensator.hpp
│   ├── detect # 识别与位姿解算
│   │   ├── armor_detect
│   │   │   ├── armor_detector_base.hpp
│   │   │   ├── armor_detector_ncnn.hpp
│   │   │   ├── armor_detector_ncnn_wrapper.hpp
│   │   │   ├── armor_detector_opencv.hpp
│   │   │   ├── armor_detector_opencv_wrapper.hpp
│   │   │   ├── armor_detector_openvino.hpp
│   │   │   ├── armor_detector_openvino_wrapper.hpp
│   │   │   ├── armor_detector_trt.hpp
│   │   │   ├── armor_detector_trt_wrapper.hpp
│   │   │   ├── armor_pose_estimator.hpp
│   │   │   └── light_corner_corrector.hpp
│   │   ├── ba_solver.hpp
│   │   ├── detector_factory.hpp
│   │   ├── graph_optimizer.hpp
│   │   ├── mono_measure_tool.hpp
│   │   ├── pnp_solver.hpp
│   │   └── rune_detect
│   │       ├── rune_detector_base.hpp
│   │       ├── rune_detector_ncnn.hpp
│   │       ├── rune_detector_ncnn_wrapper.hpp
│   │       ├── rune_detector_openvino.hpp
│   │       ├── rune_detector_openvino_wrapper.hpp
│   │       ├── rune_detector_trt.hpp
│   │       └── rune_detector_trt_wrapper.hpp
│   ├── driver # 驱动
│   │   ├── crc8_crc16.hpp
│   │   ├── hik.hpp
│   │   ├── packet_typedef.hpp
│   │   ├── serial.hpp
│   │   ├── serial_type.hpp
│   │   ├── sharetype.hpp
│   │   └── tools
│   │       ├── labeler.hpp
│   │       ├── recorder.hpp
│   │       └── video_player.hpp
│   ├── tracker # 跟踪/预测器
│   │   ├── math # 数学
│   │   │   ├── curve_fitter.hpp
│   │   │   ├── error_state_extended_kalman_filter.hpp
│   │   │   └── extended_kalman_filter.hpp
│   │   ├── motion_models # 运动模型
│   │   │   ├── motion_modela.hpp
│   │   │   ├── motion_modelonea.hpp
│   │   │   ├── motion_modeloneca.hpp
│   │   │   ├── motion_modeloneypd.hpp
│   │   │   ├── motion_modelr.hpp
│   │   │   ├── motion_modelrypd.hpp
│   │   │   └── motion_modelypd.hpp
│   │   ├── one_ca_tracker.hpp
│   │   ├── one_tracker.hpp
│   │   ├── one_ypd_tracker.hpp
│   │   ├── tracker.hpp
│   │   ├── tracker_manager.hpp
│   │   └── ypd_tracker.hpp
│   ├── type # 数据类型
│   │   ├── image.hpp
│   │   └── type.hpp
│   └── wust_vision.hpp
├── model # 模型
│   ├── label.txt
│   ├── lenet.onnx
│   ├── mlp.onnx
│   ├── opt-1208-001.bin
│   ├── opt-1208-001.engine
│   ├── opt-1208-001.onnx
│   ├── opt-1208-001.param
│   ├── yolox_rune_3.6m.bin
│   ├── yolox_rune_3.6m.engine
│   ├── yolox_rune_3.6m_ncnn.bin
│   ├── yolox_rune_3.6m_ncnn.param
│   ├── yolox_rune_3.6m.onnx
│   ├── yolox_rune_3.6m.xml
│   ├── yolox_rune.bin
│   ├── yolox_rune.engine
│   ├── yolox_rune_ncnn.bin
│   ├── yolox_rune_ncnn.param
│   ├── yolox_rune.onnx
│   └── yolox_rune.xml
├── README.md
├── run.sh
├── src
│   ├── common # 通用
│   │   ├── calculation.cpp
│   │   ├── debug # 调试
│   │   │   ├── matplottools.cpp
│   │   │   ├── tools.cpp
│   │   │   └── toolsgobal.cpp
│   │   ├── gobal.cpp
│   │   └── utils.cpp
│   ├── control # 控制
│   │   ├── armor_solver.cpp
│   │   ├── manual_compensator.cpp
│   │   ├── rune_solver.cpp
│   │   └── trajectory_compensator.cpp
│   ├── detect # 识别与位姿解算
│   │   ├── armor_detect
│   │   │   ├── armor_detector_ncnn.cpp
│   │   │   ├── armor_detector_ncnn_wrapper.cpp
│   │   │   ├── armor_detector_opencv.cpp
│   │   │   ├── armor_detector_opencv_wrapper.cpp
│   │   │   ├── armor_detector_openvino.cpp
│   │   │   ├── armor_detector_openvino_wrapper.cpp
│   │   │   ├── armor_detector_trt.cpp
│   │   │   ├── armor_detector_trt_wrapper.cpp
│   │   │   ├── armor_pose_estimator.cpp
│   │   │   └── light_corner_corrector.cpp
│   │   ├── ba_solver.cpp
│   │   ├── graph_optimizer.cpp
│   │   ├── mono_measure_tool.cpp
│   │   ├── pnp_solver.cpp
│   │   └── rune_detect
│   │       ├── rune_detector_ncnn.cpp
│   │       ├── rune_detector_ncnn_wrapper.cpp
│   │       ├── rune_detector_openvino.cpp
│   │       ├── rune_detector_openvino_wrapper.cpp
│   │       ├── rune_detector_trt.cpp
│   │       └── rune_detector_trt_wrapper.cpp
│   ├── driver # 驱动
│   │   ├── crc8_crc16.cpp
│   │   ├── hik.cpp
│   │   ├── serial.cpp
│   │   └── tools
│   │       ├── labeler.cpp
│   │       ├── recorder.cpp
│   │       └── video_player.cpp
│   ├── main.cpp
│   ├── tracker # 跟踪/预测器
│   │   ├── math # 数学
│   │   │   ├── curve_fitter.cpp
│   │   │   ├── error_state_extended_kalman_filter.cpp
│   │   │   └── extended_kalman_filter.cpp
│   │   ├── one_ca_tracker.cpp
│   │   ├── one_tracker.cpp
│   │   ├── one_ypd_tracker.cpp
│   │   ├── tracker.cpp
│   │   ├── tracker_manager.cpp
│   │   └── ypd_tracker.cpp
│   └── wust_vision.cpp
├── static # web使用的静态文件
│   └── logo.JPG
├── templates # web模板
│   └── index.html
├── video.py 
└── web.py 
```
## 本项目的优势
* 识别推理基于线程池，帧率取决于相机的取流帧率，即使使用神经网络（推理时间大约15ms左右）在nuc与orin nx测试，帧率依然稳定160hz+，基于传统算法进行优化，精度较神经网络高，而使用神经网络鲁棒性比传统算法高
* 使用ypd作为ekf的观测量，噪声更小，调参简单，对角度观测维度特殊处理，残差计算更合理与稳定，使用高斯牛顿迭代，非线性预测效果更好（本项目还有iesekf的实现，但rm工况下效果不如iekf）
* 