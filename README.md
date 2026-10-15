# TCP-PPO

This is the official implementation of the paper **"Temporal Convolutional Processing with Proximal Policy Optimization for Reliable Autonomous Navigation in Magnetically Controlled Capsule Endoscopy"**. It is built on Unity [ML-Agents](https://github.com/Unity-Technologies/ml-agents) and evaluated in the VR-Caps simulator.

## Overview

Covering the whole examination area is a major challenge in Magnetically Controlled Capsule Endoscopy (MCCE). We model navigation as a POMDP: the agent sees the capsule and magnet head states but must remember what it has already covered.

TCP-PPO adds a CNN-based temporal encoder to PPO. The encoder processes the last *m* states, so the policy can use patterns from its past trajectory. A dual-component reward combines two signals:

- **State reward:** immediate coverage gain.
- **Path reward:** average coverage progress over the window, which discourages oscillating and revisiting areas.

## Key Points

- **First POMDP formulation** of MCCE navigation with temporal processing.
- **Controlled ablation:** compared against LSTM-PPO, GRU-PPO and non-temporal PPO.
- **Efficient:** the CNN encoder uses fewer parameters than the recurrent encoders and has sub-millisecond inference.
- **Rigorous evaluation:** five independently trained seeds per architecture.
