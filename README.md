# OpenArm-Gazebo-RL
## 本仓库主要记录基于强化学习的OpenArm或Gazebo的相关任务

### 2026.8.11 上传基于OpenArm-Gazebo的非预设轨迹强化学习轴孔装配任务v1

**v1版本贡献：**

SAC算法植入SEARCHING阶段，可以跑通，参数未优化。

**v2版本贡献：**

（1）md文件中针对SEARCHING阶段加入了MIT控制的相关说明方便理解，并加入每次修改参数后重新编译的指令。

（2）单独将SEARCHING阶段搜索最大步长从0.2mm分别调整为0.025mm、0.05mm进行验证，抖动情况均相比0.2mm时稍有改善，但0.025mm步长较短，出现了大量timeout的情况，因此v2最终采用了0.05mm作为搜索步长。

（3）针对（2）仍出现的情况，v2将调整SEARCHING阶段的位置刚度、阻尼等参数。

（4）在train过程中加入了记录每个Episode的action日志。
