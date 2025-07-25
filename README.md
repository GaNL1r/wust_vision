# <img src="https://s21.ax1x.com/2025/07/01/pVu3xA0.md.jpg" width="30">WUST_VISION



## 依赖
* OpenCV
* [OpenVINO](https://flowus.cn/7a2a3341-74a1-4db9-bced-99fe5d05ab75)/[TensorRT-cuda](https://flowus.cn/e98af178-de0b-4546-808d-a6f1ff199d62)/[NCNN](https://flowus.cn/664f6bee-8ea9-4d54-8a78-e2c0bf38ee9f)/[OnnxRunetime](https://flowus.cn/8fbecbbf-c0f9-49bb-bac5-7b4923f55c99)连接为简单部署文档
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
git clone --recurse-submodules https://github.com/WUST-RM/wust_vision.git
cd wust_vision
sudo ./run.sh run/rebuild/build/cal #编译并运行/删除build缓存重新编译/仅编译/简单相机标定
```
### 注意：本项目可选择编译OpenVINO/TensorRT-cuda/NCNN/OnnxRunetime与能量机关，需在build缓存前在[CMakeLists.txt](CMakeLists.txt)中修改对应编译选项,修改后需rebuild重新编译，无OpenVINO/TensorRT-cuda/NCNN/OnnxRunetime环境仍可以使用OpenCV的装甲板识别，装甲板/能量机关的识别方案需要在[config/config_common.yaml](config/config_common.yaml)中修改
## 文件树
```
.
├── CMakeLists.txt
├── cmake_modules
│   ├── FindG2O.cmake
│   ├── FindHikSDK.cmake
│   ├── FindOrt.cmake
│   └── FindTensorRT.cmake
├── config
│   ├── 7.9-600-20s-7.5rad-60-113.yaml
│   ├── armor_detect_opencv.yaml
│   ├── camera_calibrator.yaml
│   ├── camera_info.yaml
│   ├── config_common.yaml
│   ├── detect_ncnn.yaml
│   ├── detect_openvino.yaml
│   ├── detect_ort.yaml
│   ├── detect_trt.yaml
│   ├── guard.sh
│   └── omni_config.yaml
├── dandao.py
├── format.sh
├── include
│   ├── common
│   │   ├── 3rdparty
│   │   │   ├── angles.h
│   │   │   └── matplotlibcpp.h
│   │   ├── calculation.hpp
│   │   ├── camera_calibrator.hpp
│   │   ├── debug
│   │   │   ├── matplottools.hpp
│   │   │   ├── toolsgobal.hpp
│   │   │   └── tools.hpp
│   │   ├── gobal.hpp
│   │   ├── logger.hpp
│   │   ├── tf.hpp
│   │   ├── ThreadPool.h
│   │   └── utils.hpp
│   ├── control
│   │   ├── armor_solver.hpp
│   │   ├── manual_compensator.hpp
│   │   ├── rune_solver.hpp
│   │   └── trajectory_compensator.hpp
│   ├── detect
│   │   ├── armor_detect
│   │   │   ├── armor_detect_common.hpp
│   │   │   ├── armor_detector_base.hpp
│   │   │   ├── armor_detector_opencv.hpp
│   │   │   ├── armor_detector_opencv_wrapper.hpp
│   │   │   ├── armor_pose_estimator.hpp
│   │   │   ├── light_corner_corrector.hpp
│   │   │   ├── ncnn
│   │   │   │   ├── armor_detector_ncnn.hpp
│   │   │   │   └── armor_detector_ncnn_wrapper.hpp
│   │   │   ├── number_classifier.hpp
│   │   │   ├── onnxruntime
│   │   │   │   ├── armor_detector_ort.hpp
│   │   │   │   └── armor_detector_ort_wrapper.hpp
│   │   │   ├── openvino
│   │   │   │   ├── armor_detector_openvino.hpp
│   │   │   │   └── armor_detector_openvino_wrapper.hpp
│   │   │   └── tensorrt
│   │   │       ├── armor_detector_trt.hpp
│   │   │       └── armor_detector_trt_wrapper.hpp
│   │   ├── ba_solver.hpp
│   │   ├── detector_factory.hpp
│   │   ├── graph_optimizer.hpp
│   │   ├── mono_measure_tool.hpp
│   │   ├── pnp_solver.hpp
│   │   └── rune_detect
│   │       ├── ncnn
│   │       │   ├── rune_detector_ncnn.hpp
│   │       │   └── rune_detector_ncnn_wrapper.hpp
│   │       ├── onnxruntime
│   │       │   ├── rune_detector_ort.hpp
│   │       │   └── rune_detector_ort_wrapper.hpp
│   │       ├── openvino
│   │       │   ├── rune_detector_openvino.hpp
│   │       │   └── rune_detector_openvino_wrapper.hpp
│   │       ├── rune_detector_base.hpp
│   │       └── tensorrt
│   │           ├── rune_detector_trt.hpp
│   │           └── rune_detector_trt_wrapper.hpp
│   ├── driver
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
│   ├── omni.hpp
│   ├── tracker
│   │   ├── curve_fitter.hpp
│   │   ├── motion_models
│   │   │   ├── acc_model.hpp
│   │   │   ├── motion_modela.hpp
│   │   │   ├── motion_modelonea.hpp
│   │   │   ├── motion_modeloneca.hpp
│   │   │   ├── motion_modeloneypd.hpp
│   │   │   ├── motion_modelr.hpp
│   │   │   ├── motion_modelrypd.hpp
│   │   │   └── motion_modelypd.hpp
│   │   ├── one_ca_tracker.hpp
│   │   ├── one_tracker.hpp
│   │   ├── tracker.hpp
│   │   └── tracker_manager.hpp
│   ├── type
│   │   ├── image.hpp
│   │   └── type.hpp
│   └── wust_vision.hpp
├── KalmanHyLib
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── 3rdparty
│   │   │   └── angles.h
│   │   └── KalmanHyLib
│   │       ├── adaptive_extended_kalman_filter.hpp
│   │       ├── error_state_extended_kalman_filter.hpp
│   │       ├── extended_kalman_filter.hpp
│   │       ├── kalman_hybird_lib.hpp
│   │       └── unscented_kalman_filter.hpp
│   └── README.md
├── model
│
├── README.md
├── run.sh
├── src
│   ├── camera_calibrator_main.cpp
│   ├── common
│   │   ├── calculation.cpp
│   │   ├── camera_calibrator.cpp
│   │   ├── debug
│   │   │   ├── matplottools.cpp
│   │   │   ├── tools.cpp
│   │   │   └── toolsgobal.cpp
│   │   └── gobal.cpp
│   ├── control
│   │   ├── armor_solver.cpp
│   │   ├── manual_compensator.cpp
│   │   ├── rune_solver.cpp
│   │   └── trajectory_compensator.cpp
│   ├── detect
│   │   ├── armor_detect
│   │   │   ├── armor_detect_common.cpp
│   │   │   ├── armor_detector_opencv.cpp
│   │   │   ├── armor_detector_opencv_wrapper.cpp
│   │   │   ├── armor_pose_estimator.cpp
│   │   │   ├── light_corner_corrector.cpp
│   │   │   ├── ncnn
│   │   │   │   ├── armor_detector_ncnn.cpp
│   │   │   │   └── armor_detector_ncnn_wrapper.cpp
│   │   │   ├── number_classifier.cpp
│   │   │   ├── onnxruntime
│   │   │   │   ├── armor_detector_ort.cpp
│   │   │   │   └── armor_detector_ort_wrapper.cpp
│   │   │   ├── openvino
│   │   │   │   ├── armor_detector_openvino.cpp
│   │   │   │   └── armor_detector_openvino_wrapper.cpp
│   │   │   └── tensorrt
│   │   │       ├── armor_detector_trt.cpp
│   │   │       └── armor_detector_trt_wrapper.cpp
│   │   ├── ba_solver.cpp
│   │   ├── graph_optimizer.cpp
│   │   ├── mono_measure_tool.cpp
│   │   ├── pnp_solver.cpp
│   │   └── rune_detect
│   │       ├── ncnn
│   │       │   ├── rune_detector_ncnn.cpp
│   │       │   └── rune_detector_ncnn_wrapper.cpp
│   │       ├── onnxruntime
│   │       │   ├── rune_detector_ort.cpp
│   │       │   └── rune_detector_ort_wrapper.cpp
│   │       ├── openvino
│   │       │   ├── rune_detector_openvino.cpp
│   │       │   └── rune_detector_openvino_wrapper.cpp
│   │       └── tensorrt
│   │           ├── rune_detector_trt.cpp
│   │           └── rune_detector_trt_wrapper.cpp
│   ├── driver
│   │   ├── crc8_crc16.cpp
│   │   ├── hik.cpp
│   │   ├── serial.cpp
│   │   └── tools
│   │       ├── labeler.cpp
│   │       ├── recorder.cpp
│   │       └── video_player.cpp
│   ├── main.cpp
│   ├── omni.cpp
│   ├── tracker
│   │   ├── curve_fitter.cpp
│   │   ├── one_ca_tracker.cpp
│   │   ├── one_tracker.cpp
│   │   ├── tracker.cpp
│   │   └── tracker_manager.cpp
│   └── wust_vision.cpp
├── static
│   └── logo.JPG
├── templates
│   └── index.html
├── video.py
└── web.py



```
## 性能
* 由于使用线程池，理论处理帧率完全取决于图像的取流帧率，几乎不受处理时长的影响（由于线程池加入了动态分配上限的机制，如果处理速度过慢，则帧率可能达不到预期，本项目在正常部署后不会出现这种问题），对于1440*1080的图像，经过测试在12代nuc openvino版本下平均处理时长为10-15ms（包括能量机关），opencv版本为1-4ms，在jeson orin nx 8g tensorrt版本为5-10ms（包括能量机关），opencv版本为1-4ms
* 对相机采集线程进行高优先级处理，使得相机帧率尽可能达到最高，进而提升处理帧率
* 装甲板识别基于4点模型的神经网络，鲁棒性高，同时通过cv算法提升精度，而cv算法取不到解的情况神经网络的精度也足够弥补这种问题
* 配套网页调试器（开启debug，启动 [web.py](web.py) ），可视化内容不在部署机渲染，在debug下仍然有较高的性能，同时也无需接入键鼠和屏幕
* 对于坐标变化采取了简化思路，只更新电控发来的gimbal到odom的rpy的变换，输入缓冲区缓存，在相机取流时查找曝光时间一半减通信延迟缓冲区并插值构造R_gimbal2odom然后通过静态变换构造T_camera2odom，随图像传到位姿解算，得到精准的装甲板在odom的位姿
* 独立日志库，log可显示代码位置与触发时间
* 采用ekf与esekf的融合，同时针对角度的观测维度做了特殊处理，拥有整车模型和单装甲板模型，使得预测与击打更加鲁棒
* 支持参数动态加载