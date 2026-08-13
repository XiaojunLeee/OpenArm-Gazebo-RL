# OpenArm Gazebo YZ Spiral Hole Search

## SAC 搜索路径（V1）

原 `hole_search_sim.launch.py` 与固定螺旋控制器保持不变。新增
`sac_hole_search_sim.launch.py` 仅把 `SEARCHING` 的 Y/Z 螺旋参考替换为
SAC 每 0.05 个仿真秒的一次二维位置增量；`HOLDING`、`APPROACHING`、
`INSERTING`、`INSERT_HOLDING` 仍调用原规则控制。

启动仿真：

```bash
cd ~/openarm_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch openarm_gazebo_hole_search sac_hole_search_sim.launch.py rviz:=false
```

另开终端，进入 `openarm` Conda 环境后训练：

```bash
cd ~/openarm_ws/src/openarm_gazebo_hole_search_raw
conda run -n openarm env PYTHONPATH=$PWD python scripts/train_sac.py
```

训练终端会逐回合打印编号、结果和结束原因；每 250 回合打印并保存检查点、
验证指标和最新最佳模型来源。产物位置：`sac_results/models/`；TensorBoard
日志位置：`sac_results/tensorboard/`。训练完成后以固定 200 孔测试：

```bash
conda run -n openarm env PYTHONPATH=$PWD python scripts/test_sac.py
```

This ROS 2 package simulates a right OpenArm performing a spiral search in the
world YZ plane while applying a constant force along world X. Gazebo provides
rigid-body dynamics and contact; RViz displays the robot, target point, tool
point, force arrow, and controller state.

The arm uses the same right-arm side-mount transform as the OpenArm v1
description (`xyz="0 -0.031 0.698"`, `rpy="1.5708 0 0"`).

## Build

```bash
cd ~/openarm_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select openarm_description openarm_gazebo_hole_search \
  --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch openarm_gazebo_hole_search hole_search_sim.launch.py
```

Disable RViz if only Gazebo is needed:

```bash
ros2 launch openarm_gazebo_hole_search hole_search_sim.launch.py rviz:=false
```

The controller starts after the model and controllers are spawned, holds the
initial YZ point for 0.5 seconds, ramps the X force over one second, waits
for confirmed tool contact, and then expands the YZ spiral. It publishes
effort commands to:

```text
/arm_effort_controller/commands
```

Parameters are in `config/spiral_search.yaml`. The most relevant are `fx`,
`r_max`, `r_rate`, `omega`, `kyz`, `dyz`, `max_x_travel`,
`hole_entry_depth`, `insert_target_depth`, `insert_max_depth`, and
`max_search_time`.

The fixture consists of four collision boxes around a 7 mm square through
hole, used as a conservative approximation of the approximately 7 mm circular
lock hole. The simulated fixture and hole passage are 17 mm deep along X,
matching the measured lock-hole depth. Its visuals are partially transparent
so insertion can be inspected directly in Gazebo without changing the
collision geometry. The search tool is a 6 mm diameter key/pin. After detecting hole entry, the
simulation controller moves the YZ reference to the known hole center and
inserts to `insert_target_depth`. It then captures all seven joint angles,
removes the Cartesian search wrench, and remains in `INSERT_HOLDING` using
MIT-equivalent joint torque control:
`tau = ramp*Kp*(q_hold-q) - Kd*dq + gravity`.
The target insertion depth must remain satisfied for
`insert_hold_confirm_time` before the joint angles are captured. No 2 mm
center-distance condition is required. The existing
`hole_center_clearance` check only represents whether the 6 mm pin physically
fits inside the 7 mm opening (0.5 mm center-position clearance); this prevents
contact-solver penetration into the solid plate from being accepted as a
successful insertion.
The filtered joint and tool speeds must also be below
`insert_hold_max_joint_velocity` and `insert_hold_max_tool_velocity` during
that confirmation window. This captures the inserted pose after motion has
settled instead of while the pin is still moving; it does not add a tighter
YZ-position requirement.
During insertion, `insert_dx` reduces the axial push when the pin moves into
the hole too quickly and increases it during rebound, up to
`max_insert_x_force`. The search phase still uses the configured constant
world-X force.
The position gain ramps in over `mit_hold_ramp_time`, and simulated joint
velocity feedback is low-pass filtered by `mit_velocity_filter_tau`. Hold logs
also report `inside_hole`; this is diagnostic only and is not an additional
condition for entering joint hold. Its display uses
`hole_inside_hysteresis` to avoid repeatedly toggling at the physical
clearance boundary.
To prevent a small joint-space error from becoming a large tool-tip motion,
the simulation adds a bounded Cartesian restoring wrench around the captured
inserted pose and maps it to MIT feed-forward joint torque. Its gains are
`mit_hold_cartesian_kx`, `mit_hold_cartesian_dx`,
`mit_hold_cartesian_kyz`, and `mit_hold_cartesian_dyz`.
The supplied Gazebo gains are joint-side values and are therefore higher than
the motor-side MIT gains used by the hardware executable; a direct numerical
copy would omit the transmission's effective stiffness.
`mit_hold_max_joint_error` and `mit_hold_max_joint_velocity` provide hold
safety limits after the initial `mit_hold_safety_delay` braking window.
Joint-limit violations must persist for `mit_hold_fault_confirm_time` before
entering `FAULT`, so a single Gazebo contact-solver velocity spike does not
trip the controller. It does not return to `INSERTING` when contact dynamics
cause a transient depth error. The insertion phase uses stronger YZ centering gains
(`insert_kyz`, `insert_dyz`, and `max_insert_yz_force`).
X-travel and maximum insertion-depth protections remain active.

The controller publishes a static `world -> hole_center` transform at the
fixture front-face center. The supplied RViz configuration enables the TF
display, so the hole coordinates and axes are visible directly.

The alternate default test point places the hole approximately 1.65 mm along
-Y and 1.00 mm along +Z from the nominal initial tool axis, giving about
1.93 mm total radial offset. This exercises both Y and Z search motion and
leaves the pin about 1.43 mm outside the 0.5 mm effective insertion clearance,
so the search starts near the hole without starting already inside it. The
fixture front face has only about 0.05 mm nominal
overlap with the initial tool-tip X. This is enough for Gazebo to report
startup contact without the large contact impulse caused by the previous
0.5 mm penetration. Because the physical clearance is small, the default
spiral is limited to 4 mm and its radial growth is reduced
to 0.2 mm/s; at 2 rad/s its turn-to-turn pitch is about 0.63 mm.
The YZ search center remains the startup tool position after contact
confirmation, rather than being recaptured from a contact-deflected sample.
The supplied YZ and joint damping values are increased for the much tighter
6/7 mm contact geometry. A bounded rotational impedance
(`orientation_k`, `orientation_d`, and `max_orientation_torque`) also holds
the startup tool orientation while approaching, searching, and inserting; this
prevents the long pin from tilting and amplifying small joint motion into a
large tip displacement.

The 6 mm pin keeps its true visual size, while its rigid Gazebo collision is
disabled. Plate contact is instead represented by a bounded virtual
spring-damper using `virtual_contact_k`, `virtual_contact_d`, and
`max_virtual_contact_force`. The reaction is active while the pin center is
outside the 0.5 mm insertion clearance and is removed only when the pin is
aligned with the hole. This hybrid model retains Gazebo joint dynamics and
effort/MIT-style control, but avoids DART's unstable impulses at the sharp
edges of a sub-millimetre clearance. Set `use_virtual_plate_contact` to
`false` only if a different physical contact model is supplied.
In virtual-contact mode, entering the physical 0.5 mm center clearance at the
plate surface immediately starts `INSERTING`; requiring 3 mm depth before
removing the virtual plate reaction would make insertion impossible.
The default simulated X preload is 1 N. This is deliberately lower than the
earlier 3 N setting because the zero-gravity Gazebo arm has much less passive
dissipation than the real transmission; joint and Cartesian velocity damping
are correspondingly higher.
After hole alignment, `insert_fx` raises the axial insertion force to 2 N so
the damped model can reach the measured 17 mm lock-hole depth. The independent
`insert_max_depth` over-travel guard is 20 mm, leaving 3 mm of simulation
safety margin beyond the nominal seated depth. Once alignment has been accepted,
joint capture checks insertion depth and settling velocity but does not apply
a second YZ-distance gate.
The search-phase spring toward the initial joint posture is disabled only
during `INSERTING`; otherwise it opposes the axial force and creates a false
equilibrium after a few millimetres. Joint damping, YZ centering, tool
orientation control, torque limits, travel protection, and depth protection
remain active throughout insertion.
For this simulation the OpenArm description macro is called with
`joint_damping="2.0"`, so passive damping is evaluated synchronously by the
Gazebo physics engine. The macro parameter defaults to zero, leaving all other
OpenArm description users unchanged.

Because this is a deterministic simulation, hole entry is confirmed from both
the X insertion relative to the known fixture front surface
(`hole_surface_x`) and the known hole-center clearance (`hole_center_y`,
`hole_center_z`, and `hole_center_clearance`). This avoids treating a
collision-rebound-dependent contact sample as the depth origin. On hardware,
replace this geometric confirmation with the force/position signature from
the real force sensor.

The supplied world uses zero gravity so the arm cannot fall during the short
gap between Gazebo controller activation and the controller's first effort
message. Rigid-body inertia, collision, friction, and contact dynamics remain
active. To experiment with gravity, set world gravity to `0 0 -9.81`, set
`gravity_scale` to `1.0`, and add a position-hold startup controller before
switching to effort control.

## Safety and scope

This package never opens a CAN interface. Its controller is simulation-only.
The parameters are starting values for Gazebo and must not be copied directly
to hardware without hardware-specific safety review.

## SAC 实时日志与自动断点续训

从功能包根目录使用以下脚本启动：

```bash
bash scripts/run_sac_sim_logged.sh
bash scripts/run_sac_training_logged.sh
```

日志分别实时写入：

```text
sac_results/logs/simulation.log
sac_results/logs/training.log
```

每次启动只清空对应的旧日志，不会删除模型、Replay Buffer、检查点或
TensorBoard 数据。训练每 100 个完整 Episode 保存并在 30 个固定孔上验证。
按一次 `Ctrl+C` 后，训练脚本会把最近完整 Episode 对应的模型、Replay Buffer、
训练进度和配置保存到：

```text
sac_results/models/interrupt_checkpoint
```

下一次运行训练启动脚本时，会自动比较周期检查点与中断检查点，从最新的兼容
状态继续；若要从头训练，请先手动删除整个 `sac_results` 目录。
