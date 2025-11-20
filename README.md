# <img src="https://s21.ax1x.com/2025/08/12/pVwPPKS.png" width="40">WUST_VISION
武汉科技大学崇实战队视觉代码仓库

## 写在前面
本项目基于[中南大学FYT战队2024赛季视觉框架开源](https://github.com/CSU-FYT-Vision/FYT2024_vision),华南师范大学PIONEER战队@chenjunnn[rm_vision](https://github.com/chenjunnn/rm_vision)，华南理工大学 华南虎战队[rm_vision_core](https://github.com/scutrobotlab/rm_vision_core)（本项目的能量机关识别部分完全使用了rm_vision_core的源码，未保留在本仓库）修改与适配，参考了深圳北理莫斯科大学北极熊战队/四川大学火锅战队/沈阳航空航天大学TUP战队/北京科技大学Reborn战队/同济大学superpower战队的部分代码与模型，感谢以上开源为本队以及本人的帮助

## 依赖
* [wust_vl](https://github.com/WUST-RM/wust_vl)
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
sudo ./run.sh run xx /rebuild/build #编译并运行xx可执行文件/删除build缓存重新编译/仅编译
```
### 注意：本项目可选择编译OpenVINO/TensorRT-cuda/NCNN/OnnxRunetime，需在build缓存前在[CMakeLists.txt](CMakeLists.txt)中修改对应编译选项,修改后需rebuild重新编译，无OpenVINO/TensorRT-cuda/NCNN/OnnxRunetime环境仍可以使用OpenCV的装甲板识别，装甲板的识别方案需要在[config/auto_aim.yaml](config/auto_aim.yaml)中修改
## 文件树
```
.
├── 3rdparty
│   ├── angles.h
│   └── backward-cpp
│       
├── calibration
│   ├── calcamera.cpp
│   ├── capture.cpp
│   └── eye_hand.cpp
├── cmake
│   ├── FindG2O.cmake
│   ├── FindHikSDK.cmake
│   ├── FindOrt.cmake
│   └── FindTensorRT.cmake
├── CMakeLists.txt
├── config
│   ├── auto_aim.yaml
│   ├── auto_buff.yaml
│   ├── cal.yaml
│   ├── camera_calibrator.yaml
│   ├── camera_info.yaml
│   ├── camera.yaml
│   ├── common.yaml
│   ├── detect_ncnn.yaml
│   ├── detect_opencv.yaml
│   ├── detect_openvino.yaml
│   ├── detect_ort.yaml
│   ├── detect_trt.yaml
│   └── guard.sh
├── cuda_infer
│   ├── armor_infer.cu
│   ├── armor_infer.hpp
│   ├── CMakeLists.txt
│   ├── letter_box.cu
│   ├── letter_box.hpp
│   ├── rune_infer.cu
│   └── rune_infer.hpp
├── dandao.py
├── env.bash
├── format.sh
├── fun
│
├── KalmanHyLib
│   ├── adaptive_extended_kalman_filter.hpp
│   ├── error_state_extended_kalman_filter.hpp
│   ├── extended_kalman_filter.hpp
│   ├── gtsam.hpp
│   ├── kalman_hybird_lib.hpp
│   ├── kf.hpp
│   └── unscented_kalman_filter.hpp
├── model
│
├── README.md
├── ros2
│   ├── CMakeLists.txt
│   ├── ros2.cpp
│   └── ros2.hpp
├── run.sh
├── src
│   ├── sentry.cpp
│   └── standard.cpp
├── static
│   ├── 崇实战队logo图标.png
│   ├── css
│   │   └── style.css
│   ├── js
│   │   ├── chart_logic.js
│   │   ├── json_view.js
│   │   └── main.js
│   └── logo.JPG
├── tasks
│   ├── auto_aim
│   │   ├── armor_control
│   │   │   ├── aimer.cpp
│   │   │   ├── aimer.hpp
│   │   │   ├── planner.cpp
│   │   │   ├── planner.hpp
│   │   │   ├── shooter.cpp
│   │   │   ├── shooter.hpp
│   │   │   └── tinympc
│   │   │       
│   │   ├── armor_detect
│   │   │   ├── armor_detect_common.cpp
│   │   │   ├── armor_detect_common.hpp
│   │   │   ├── armor_detector_base.hpp
│   │   │   ├── armor_infer.cpp
│   │   │   ├── armor_infer.hpp
│   │   │   ├── armor_pose_estimator.cpp
│   │   │   ├── armor_pose_estimator.hpp
│   │   │   ├── detector_factory.hpp
│   │   │   ├── light_corner_corrector.cpp
│   │   │   ├── light_corner_corrector.hpp
│   │   │   ├── ncnn
│   │   │   │   ├── armor_detector_ncnn.cpp
│   │   │   │   ├── armor_detector_ncnn.hpp
│   │   │   │   ├── armor_detector_ncnn_wrapper.cpp
│   │   │   │   └── armor_detector_ncnn_wrapper.hpp
│   │   │   ├── number_classifier.cpp
│   │   │   ├── number_classifier.hpp
│   │   │   ├── onnxruntime
│   │   │   │   ├── armor_detector_ort.cpp
│   │   │   │   ├── armor_detector_ort.hpp
│   │   │   │   ├── armor_detector_ort_wrapper.cpp
│   │   │   │   └── armor_detector_ort_wrapper.hpp
│   │   │   ├── opencv
│   │   │   │   ├── armor_detector_opencv.cpp
│   │   │   │   ├── armor_detector_opencv.hpp
│   │   │   │   ├── armor_detector_opencv_wrapper.cpp
│   │   │   │   └── armor_detector_opencv_wrapper.hpp
│   │   │   ├── openvino
│   │   │   │   ├── armor_detector_openvino.cpp
│   │   │   │   ├── armor_detector_openvino.hpp
│   │   │   │   ├── armor_detector_openvino_wrapper.cpp
│   │   │   │   └── armor_detector_openvino_wrapper.hpp
│   │   │   └── tensorrt
│   │   │       ├── armor_detector_trt.cpp
│   │   │       ├── armor_detector_trt.hpp
│   │   │       ├── armor_detector_trt_wrapper.cpp
│   │   │       └── armor_detector_trt_wrapper.hpp
│   │   ├── armor_optimize
│   │   │   ├── ba_solver.cpp
│   │   │   ├── ba_solver.hpp
│   │   │   ├── graph_optimizer.cpp
│   │   │   └── graph_optimizer.hpp
│   │   ├── armor_tracker
│   │   │   ├── motion_models
│   │   │   │   ├── acc_model.hpp
│   │   │   │   ├── factorypd.hpp
│   │   │   │   ├── motion_modela.hpp
│   │   │   │   ├── motion_modelonea.hpp
│   │   │   │   ├── motion_modeloneca.hpp
│   │   │   │   ├── motion_modeloneypd.hpp
│   │   │   │   ├── motion_modelr.hpp
│   │   │   │   ├── motion_modelrypd.hpp
│   │   │   │   ├── motion_modelypd.hpp
│   │   │   │   └── motion_modelypdv2.hpp
│   │   │   ├── one_ca_tracker.cpp
│   │   │   ├── one_ca_tracker.hpp
│   │   │   ├── one_tracker.cpp
│   │   │   ├── one_tracker.hpp
│   │   │   ├── target.cpp
│   │   │   ├── target.hpp
│   │   │   ├── tracker.cpp
│   │   │   ├── tracker.hpp
│   │   │   ├── tracker_manager.cpp
│   │   │   ├── tracker_manager.hpp
│   │   │   ├── trackerv2.cpp
│   │   │   ├── trackerv2.hpp
│   │   │   ├── trackerv3.cpp
│   │   │   └── trackerv3.hpp
│   │   ├── auto_aim.cpp
│   │   ├── auto_aim_fsm.hpp
│   │   ├── auto_aim.hpp
│   │   ├── CMakeLists.txt
│   │   └── type.hpp
│   ├── auto_buff
│   │   ├── auto_buff.cpp
│   │   ├── auto_buff.hpp
│   │   ├── CMakeLists.txt
│   │   ├── rune_control
│   │   │   ├── aimer.cpp
│   │   │   └── aimer.hpp
│   │   ├── rune_detector
│   │   │   ├── rm_vision_core
│   │   │   ├── scut_robot_detector.cpp
│   │   │   └── scut_robot_detector.hpp
│   │   ├── rune_optimize
│   │   │   ├── ba_solver.cpp
│   │   │   ├── ba_solver.hpp
│   │   │   ├── graph_optimizer.cpp
│   │   │   └── graph_optimizer.hpp
│   │   ├── rune_tracker
│   │   │   ├── motion_models
│   │   │   │   └── motion_modelrypd.hpp
│   │   │   ├── rune_target.cpp
│   │   │   ├── rune_target.hpp
│   │   │   ├── rune_tracker.cpp
│   │   │   ├── rune_tracker.hpp
│   │   │   └── spd_fitter.hpp
│   │   └── type.hpp
│   ├── CMakeLists.txt
│   ├── debug.cpp
│   ├── debug.hpp
│   ├── packet_typedef.hpp
│   ├── type_common.hpp
│   ├── utils.hpp
│   ├── vision_base.cpp
│   └── vision_base.hpp
├── templates
│   └── index.html
├── test
│   ├── control.cpp
│   └── test_ros2.cpp
├── video.py
└── web.py





```
