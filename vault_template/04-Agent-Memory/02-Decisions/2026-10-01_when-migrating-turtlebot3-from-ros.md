---
type: architectural_constraint
status: active
created: 2026-10-01
last_reinforced: 2026-10-01
reinforcement_count: 1
confidence: 1.0
superseded_by: null
session_id: "adhoc"
tags:
  - second-brain/memory
  - memory/architectural_constraint
---

# When migrating TurtleBot3 from ROS 2 Humble/Gazebo Classic to ROS 2 Jazzy/Gazebo

> [!IMPORTANT] Learned Rule / Decision
> **Statement**: When migrating TurtleBot3 from ROS 2 Humble/Gazebo Classic to ROS 2 Jazzy/Gazebo Harmonic on WSL2: 1) WSL2 requires Ubuntu 24.04 (Noble) and apt must enforce Acquire::ForceIPv4 to avoid IPv6 unreachable connection drops. 2) Gazebo Harmonic gz-sim uses SDF 1.9 with ogre2/sensors/physics plugins and self-contained ground/sun tags to load offline in <0.2s. 3) ros_gz_bridge translates /clock, /odom, /scan (360 samples), /tf, and /cmd_vel (Twist). 4) Harmonic uses Qt Quick/QML with native Wayland/D3D12 vGPU acceleration, eliminating Gazebo Classic X11 window blanking.

### Context & Origin
- **Triggering User Prompt**: 
  > When migrating TurtleBot3 from ROS 2 Humble/Gazebo Classic to ROS 2 Jazzy/Gazebo Harmonic on WSL2: 1) WSL2 requires Ubuntu 24.04 (Noble) and apt must enforce Acquire::ForceIPv4 to avoid IPv6 unreachable connection drops. 2) Gazebo Harmonic gz-sim uses SDF 1.9 with ogre2/sensors/physics plugins and self-contained ground/sun tags to load offline in <0.2s. 3) ros_gz_bridge translates /clock, /odom, /scan (360 samples), /tf, and /cmd_vel (Twist). 4) Harmonic uses Qt Quick/QML with native Wayland/D3D12 vGPU acceleration, eliminating Gazebo Classic X11 window blanking.
- **Model Grounding**:
  Explicitly remembered via CLI with tag: ros2_jazzy_migration...

### Linked Concepts
- [[04-Agent-Memory-MOC]]
