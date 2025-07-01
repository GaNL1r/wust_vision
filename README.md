# <img src="https://s21.ax1x.com/2025/07/01/pVu3xA0.md.jpg" width="30">WUST_VISION



## 依赖
* OpenCV
* OpenVINO/TensorRT-cuda
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
make -j
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
sudo ./run.sh trt/openvino/build #TensorRT-cuda识别版本/OpenVINO识别版本/仅编译
```
### 注意：本项目默认要求OpenVINO或TensorRT-cuda参与编译（可选择其一，需在build第一次缓存前在cmakelists设置）
### OpenVINO或TensorRT-cuda实际参与装甲板与能量机关的识别，装甲板识别可使用纯OpenCV（但OpenVINO或TensorRT-cuda仍然在编译路径中），能量机关如不使用可删除