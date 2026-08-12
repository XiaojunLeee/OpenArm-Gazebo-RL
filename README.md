# OpenArm-Gazebo-RL
## 本仓库主要记录基于强化学习的OpenArm、Gazebo的相关任务

### （1）OpenArm-Gazebo非预设轨迹强化学习轴孔装配任务

#### 2026.8.11v1版本贡献：

SAC算法植入SEARCHING阶段，可以跑通，参数未优化。

#### 2026.8.12v2版本贡献：

（1）[README](https://github.com/XiaojunLeee/OpenArm-Gazebo-RL-Guideline/blob/main/v2_OpenArm-Gazebo%E9%9D%9E%E9%A2%84%E8%AE%BE%E8%BD%A8%E8%BF%B9%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BD%B4%E5%AD%94%E8%A3%85%E9%85%8D%E4%BB%BB%E5%8A%A1/README.md)文件中针对SEARCHING阶段加入了MIT控制的相关说明方便理解，并加入每次修改参数后重新编译的指令。

（2）单独将SEARCHING阶段搜索最大步长从0.2mm分别调整为0.025mm、0.05mm进行验证，抖动情况均相比0.2mm时稍有改善，但0.025mm步长较短，出现了大量timeout的情况，因此**v2最终采用了0.05mm作为搜索步长**。

（3）针对一直抖动的情况，v2调整了SEARCHING阶段的位置刚度、阻尼等参数，同时改变了SAC的参数设置。

（4）在[train](https://github.com/XiaojunLeee/OpenArm-Gazebo-RL-Guideline/blob/main/v2_OpenArm-Gazebo%E9%9D%9E%E9%A2%84%E8%AE%BE%E8%BD%A8%E8%BF%B9%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BD%B4%E5%AD%94%E8%A3%85%E9%85%8D%E4%BB%BB%E5%8A%A1/scripts/train_sac.py)过程中加入了记录每个Episode的action日志。
