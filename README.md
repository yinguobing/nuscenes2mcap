# nuscenes2mcap

将 [nuScenes](https://www.nuscenes.org/) 自动驾驶数据集转换为 **ROS2 消息格式的 MCAP 文件**。

## 概述

[nuScenes](https://www.nuscenes.org/) 是一个大型城市场景自动驾驶数据集，面向非商业用途免费提供。本项目将 nuScenes 场景数据转换为 [MCAP](https://mcap.dev/) 文件，所有消息使用 **ROS2 CDR 编码**和标准 ROS2 消息类型，可直接用于 ROS2 生态工具和 Foxglove 等可视化平台。

### 输出的 ROS2 消息类型

| ROS2 消息类型 | 数据来源 |
|---|---|
| `sensor_msgs/PointCloud2` | LIDAR（LIDAR_TOP） |
| `sensor_msgs/Image` | 六路相机（CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT） |
| `sensor_msgs/Range` | 雷达（RADAR_FRONT, RADAR_FRONT_LEFT, RADAR_FRONT_RIGHT, RADAR_BACK_LEFT, RADAR_BACK_RIGHT） |
| `sensor_msgs/NavSatFix` | GPS 位置 |
| `sensor_msgs/Imu` | IMU 数据 |
| `geometry_msgs/PoseStamped` | 自车位姿 |
| `geometry_msgs/TwistStamped` | 自车速度 |
| `geometry_msgs/Vector3Stamped` | 加速度 |
| `nav_msgs/OccupancyGrid` | 地图扩展（车道、路肩、人行道等） |
| `tf2_msgs/TFMessage` | TF 坐标变换 |
| `visualization_msgs/MarkerArray` | 场景标注（3D 框、类别标签） |
| `visualization_msgs/Marker` | CAN 总线数据（转向、刹车、油门等） |

## 使用方法

### 1. 下载数据

下载 [nuScenes mini 数据集](https://motional-nuscenes.s3.ap-northeast-1.amazonaws.com/index.html#public/)（无需登录）：

```
can_bus.zip          → data/
nuScenes-map-expansion-v1.3.zip  → data/maps/
v1.0-mini.tgz        → data/
```

### 2. 安装依赖

Python 3.11 环境：

```bash
pip install -e .
```

### 3. 运行转换

```bash
python convert_to_mcap_ros2.py
```

默认扫描 `data/` 目录下的 nuScenes 数据集，将每个场景输出为独立的 MCAP 文件，保存至 `output/` 目录。

可用参数：

| 参数 | 说明 |
|---|---|
| `--data-root PATH` | nuScenes 数据目录（默认 `data/`） |
| `--output-dir PATH` | 输出目录（默认 `output/`） |
| `--version VERSION` | 数据集版本（默认 `v1.0-mini`） |
| `--scene-id N` | 只转换指定场景 ID |
| `--max-scenes N` | 最多转换 N 个场景 |

## 依赖

- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) — 数据加载
- [rosbags](https://gitlab.com/ternaris/rosbags) — ROS2 消息定义与 CDR 序列化
- [mcap](https://github.com/foxglove/mcap) — MCAP 写入
- [pypcd](https://github.com/DanielPollithy/pypcd) — PointCloud 解析
- [Pillow](https://python-pillow.org/), `numpy`, `pyquaternion` — 数据处理

## 输出结构

每个场景生成一个 `.mcap` 文件，包含该场景所有时间步的 ROS2 消息。消息按时间戳排序，可直接用 `ros2 bag play` 播放或导入 Foxglove 查看。

## 许可

MIT License
