# Ranger 四轮转向仿真栈

本仓库整理 Ranger Mini 2.0 室内服务导航的仿真实现，涵盖任务接地、多运动序列 MPPI、四轮转向运动学执行和 Gazebo 场景接口，供理解系统结构和研究实现。实机软件见 [ranger-real-stack](https://github.com/Vitashelton/ranger-real-stack)。

## 系统链路

```text
自然语言请求 + 登记上下文 + 按需视觉
                  ↓
任务接地：服务步骤、目标、空间偏好和到达条件
                  ↓
全局参考 + 局部观测 → 多运动序列 MPPI
                  ↓    轮态预测、运动转换、代价评价与可行性复检
       运动学执行器 → Gazebo → 状态/观测反馈
                  └────────→ 滚动重规划
```

任务或视觉信息形成目标与空间条件；MPPI 根据当前轮态优化可行运动序列及转换时机。视觉语言模型不直接输出底盘控制。车体执行使用有限转向速度、切换约束与 Ranger 几何；Gazebo 的轮子为视觉模型，速度控制代理不模拟真实舵机或轮胎动力学。

## 代码结构

- `src/rangermini_dynamic_semantic/`：任务接地、视觉接口、运动学、规划器、场景启动与测试。主要任务入口为 `scripts/run_grounded_delivery.py`；目录中也保留开发期间的其他候选和对照，不能默认视为当前主方法。
- `src/ranger_sim_motion_cpp/`：C++ ROS 2 运动执行节点，复用 `integration/mppi_native_core/core.cpp`。
- `integration/mppi_native_core/`：共享运动与碰撞距离数值内核。
- `integration/nav2_mppi_adapter/`、`integration/mppi_h_adapter/`：公开 MPPI 对照的接口适配；MPPI-H 的上游源码和许可证保留在 `integration/mppi_h_upstream/`，来源和修改记录见各目录。
- `integration/ranger_ros2/`：底盘反馈接口的补丁与配置记录。
- `tools/`：任务运行、实验审计和绘图工具；部分工具依赖未公开的实验数据或场景资产。

## 实验状态与范围

已有 36 回合任务条件实验，按三个规划器、两个任务、两个上下文和三个种子组织；36 回合结束不代表全部成功。在线 Gazebo 服务回合曾包含真实语言解析、视觉响应、仿真服务确认和代理几何检查。各批次使用的代码和设置可能不同，不能混算；本仓库不包含原始日志或统计，无法仅凭仓库独立核验数值。规划候选、仿真执行与真实硬件结果必须分开解释。公开参考与本方法的接口和目标可能不同，不能仅按到站数归因算法优劣。

## 构建与使用边界

基础环境为 ROS 2 Humble、Gazebo、Nav2 和 Python 依赖；`python3 integration/mppi_native_core/build.py` 可构建共享内核，ROS 包可通过 colcon 构建。部分对照依赖 Nav2 安装库或 MPPI-H 上游依赖；具体命令参考对应适配目录。API 凭据通过环境变量提供，不写入代码。

本公开版本有意排除实验室布局与地图、第三方场景模型和检测权重、原始图像/点云、bag、实验产物、论文 PDF 及构建文件。相关场景、模型与日志引用因此需要自行准备，路径中的 `/home/USER` 和 `/home/ROBOT_USER` 也是待配置占位符。**本版本尚未在干净环境验证完整构建或端到端复现。** 不要将仿真运动学结果表述为实机、真实舵机或轮胎动力学验证。
