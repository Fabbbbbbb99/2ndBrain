---
title: "Zero-Copy cv_bridge to OpenCV Mat Recipe"
type: skill
status: active
trigger: "Processing high-frame-rate sensor_msgs/Image topics in ROS 2"
created: 2026-10-01
last_reinforced: 2026-10-01
tags:
  - second-brain/memory
  - memory/skill
---

# 🛠️ Skill: Zero-Copy cv_bridge to OpenCV Mat Recipe

> [!TIP] Execution Recipe
> **Trigger**: Processing high-frame-rate `sensor_msgs/Image` topics in ROS 2 without heap reallocation overhead.

## Step-by-Step Procedure
1. Include `<cv_bridge/cv_bridge.h>` and `<sensor_msgs/image_encodings.hpp>`.
2. In the subscription callback, use `cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)` to obtain a `CvImageConstPtr` sharing the ROS message memory buffer.
3. Access the OpenCV matrix via `cv_ptr->image`.
4. Only use `cv_bridge::toCvCopy()` if in-place pixel modification is strictly required.

---
*Linked to [[04-Agent-Memory-MOC]]*
