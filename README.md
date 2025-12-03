# MyoFullBody

**MyoFullBody – an open-source full-body musculoskeletal model**

MyoFullBody is a physiologically realistic, muscle-driven full-body model built on top of [MyoSuite](https://github.com/myohub/myosuite).  
It enables **neuromuscular motor control research**, supporting full body motion tasks with **Hill-type muscle actuators** and **no torque-based control**.

This repository also includes the **BimanualMuscle** environment for upper-body, fixed-base tasks, making the project useful for a wide range of motion imitation and control problems.

---

## Musculoskeletal Models

MyoFullBody is part of the **MuscleMimic** research project artifacts.  
We provide two complementary embodiments tailored to different applications:

- **Bimanual** – upper-body, fixed-base model
- **MyoFullBody** – full-body model

Both models are built on MyoSuite components, combining **MyoArm**, **MyoLegs**, and **MyoBack** models with Hill-type muscle actuators. This enables studying motor control at the **neuromuscular level**, rather than via idealized torque controllers.

### Environment Summary

| Model           | Type        | Joints | Muscles | Focus                        |
|-----------------|-------------|--------|---------|------------------------------|
| BimanualMuscle  | Fixed-base  | 76     | 126      | Upper-body manipulation      |
| MyoFullBody     | Full-body   | 123     | 416     | Locomotion + manipulation    |

---

## BimanualMuscle Environment

The **BimanualMuscle** environment is designed for **upper-body manipulation tas**

**Key properties**

- Up to **76 degrees of freedom**  
- Up to **126 Hill-type muscles**  
- **Collision handling**  
  - Collisions are enabled between the arms and thorax, and between the two arms


**Visualization**
TBA


## MyoFullBody Environment

The **MyoFullBody** environment provides a **comprehensive full-body musculoskeletal system** with full biomechanical detail and rich contact dynamics, suitable for **locomotion**, **manipulation**, and **whole-body imitation**.

**Key properties**

1. **Full-body kinematics**
   - **123 degrees of freedom** from pelvis to fingertips
2. **Muscle-based actuation**
   - **416 Hill-type muscle actuators** across major muscle groups  
   - Realistic activation dynamics and muscle-tendon behavior
3. **Contact-rich**
   - TBA
  
      

**Visualization**
TBA

## License
TBA

## Acknowledgements

This work is built on top of MyoSuite, which provides refinements and validation to previous models released in MyoSuite.

---
For questions, issues, or contributions, feel free to open an Issue or Pull Request on this repository.
