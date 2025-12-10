# MyoFullBody (Work In Progress)

**MyoFullBody – an open-source full-body musculoskeletal model**

MyoFullBody is part of the [**MuscleMimic**](https://github.com/amathislab/musclemimic) research project artifacts. It is a physiologically realistic, muscle-driven full-body model built on top of [MyoSuite](https://github.com/myohub/myosuite).  
It enables **neuromuscular motor control research**, supporting full body motion tasks with **Hill-type muscle actuators** and **no torque-based control**.

This repository also includes the **BimanualMuscle** environment for upper-body, fixed-base tasks, making the project useful for a wide range of motion imitation and control problems.

---

## Demos of capable motions

MyoFullBody enables realistic full-body motion control with pure muscle actuation. Below are example motions demonstrating the model's capabilities, all policies were trained with MuscleMimic.
<table>
  <tr>
    <td align="center" width="50%">
      <b>Running</b><br>
      Full-body locomotion with coordinated muscle actuation.<br>
      <video src="https://github.com/user-attachments/assets/323cdf33-95e9-41a5-a8a0-846df99c5958" width="320" controls></video>
    </td>
    <td align="center" width="50%">
      <b>Air Kick</b><br>
      Dynamic kicking motion with balance control.<br>
      <video src="https://github.com/user-attachments/assets/5198b62b-46f0-4d06-b541-6172a412cc4e" width="320" controls></video>
    </td>
  </tr>
</table>

Below we showcase a walking motion and extract the corresponding synthetic muscle activation patterns for three leg muscles along two gait cycles.
<video src="https://github.com/user-attachments/assets/da545e08-b241-451b-b02d-d3ad3e1ee41d" controls></video>
    
---

## Musculoskeletal Models

 
We provide two complementary embodiments tailored to different applications:

- **BimanualMuscle** – upper-body, fixed-base model
- **MyoFullBody** – full-body model

Both models are built on MyoSuite components, combining **MyoArm**, **MyoLegs**, and **MyoBack** models with Hill-type muscle actuators. This enables studying motor control at the **neuromuscular level**, rather than via idealized torque controllers.

### Environment Summary

| Model           | Type        | Joints | Muscles | Focus                        |
|-----------------|-------------|--------|---------|------------------------------|
| BimanualMuscle  | Fixed-base  | 76     | 126      | Upper-body manipulation      |
| MyoFullBody     | Full-body   | 123     | 416     | Locomotion + manipulation    |

---

## BimanualMuscle Environment

The **BimanualMuscle** environment is designed for **upper-body manipulation task**

**Key properties**

- Up to **76 degrees of freedom**  
- Up to **126 Hill-type muscles**  
- **Collision handling**  
  - Collisions are enabled between the arms and thorax, and between the two arms


**Visualization**
<p align="center">
  <img src="https://github.com/user-attachments/assets/cd36ec07-5fd7-4331-9f3a-a3cc6a640007"
       width="70%">
</p>


## MyoFullBody Environment

The **MyoFullBody** environment provides a **comprehensive full-body musculoskeletal system** with full biomechanical detail and rich contact dynamics, suitable for **locomotion**, **manipulation**, and **whole-body imitation**.

**Key properties**

1. **Full-body kinematics**
   - **123 degrees of freedom** from pelvis to fingertips
2. **Muscle-based actuation**
   - **416 Hill-type muscle actuators** across major muscle groups  
   - Realistic activation dynamics and muscle-tendon behavior
3. **Contact-rich**
   - We explicitly enable additional collision pairs, such as leg–leg, arm–leg, foot-foot, to capture the required self-contact behavior, including bimanual interactions. 
  
      

**Visualization**

<p align="center">
  <img src="https://github.com/user-attachments/assets/66dfaeca-4a1b-4a27-87e4-31e03101fa43"
       width="70%">
</p>

**Muscle Refinement**

While building MyoFullBody and BimanualMuscle, we corrected left–right limb asymmetries and addressed several unexpected muscle-jumping behaviors. A few representative fixes are shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c4772223-e655-46c4-9105-2db4938e6450" width="45%">
  <img src="https://github.com/user-attachments/assets/97c6645c-651a-4796-8b80-7d112ef34f91" width="45%">
</p>


**Validation**

We also cross-validate the current model using previously published cadaver and MRI data. A comprehensive analysis will be presented in our upcoming preprint, with a few illustrative examples included here.

<p align="center">
  <img src="https://github.com/user-attachments/assets/02e187ea-9257-4429-afeb-59bd5b6ff5fd" width="35%">
  <img src="https://github.com/user-attachments/assets/c3d2fe42-ebfa-49b7-b53e-302ed87b53e5" width="35%">
  <img src="https://github.com/user-attachments/assets/e2d43cda-a49f-466e-8381-7f86d608111d" width="25%">
</p>


## License
MyoFullBody is licensed under the [Apache License](https://github.com/amathislab/myofullbody/blob/main/LICENSE).

## Acknowledgements

This work is built on top of [myo_sim](https://github.com/MyoHub/myo_sim), which provides refinements and validation to previous models released in MyoSuite.

---
For questions, issues, or contributions, feel free to open an Issue or Pull Request on this repository.
