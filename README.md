# ML-Numerical-Diabetes-Glucose-ODE-Modeling

<p align="center">
  <a href="" rel="noopener">
    



</p>
<p align="center">
    <br> 
</p>
<div align="center">

[![GitHub contributors](https://img.shields.io/github/contributors/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling)](https://github.com/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling/contributors)
[![GitHub issues](https://img.shields.io/github/issues/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling)](https://github.com/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling/issues)
[![GitHub forks](https://img.shields.io/github/forks/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling)](https://github.com/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling/network)
[![GitHub stars](https://img.shields.io/github/stars/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling)](https://github.com/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling/stargazers)
[![GitHub license](https://img.shields.io/github/license/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling)](https://github.com/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling/blob/main/LICENSE)

</div>

<details>
  <summary>📚 Table of Contents</summary>
  <ol>
    <li><a href="#about">About The Project</a>
      <ul>
        <li><a href="#-built-using">Built Using</a></li>
      </ul>
    </li>
    <li><a href="#-installation">Installation</a></li>
    <li><a href="#-running-the-notebook">Running the Notebook</a></li>
    <li><a href="#-numerical-methods-for-solving-odes-in-glucose-insulin-modeling">Numerical Methods for Solving ODEs</a></li>
    <li><a href="#-whats-inside">What’s Inside (Methods)</a></li>
    <li><a href="#-key-concepts">Key Concepts (ODE)</a></li>
    <li><a href="#-methods-breakdown">Methods Breakdown</a></li>
    <li><a href="#-physics-informed-neural-network-pinn-for-glucose-insulin-dynamics">PINN for Glucose-Insulin Dynamics</a></li>
    <li><a href="#-description">PINN Description</a></li>
    <li><a href="#-whats-inside-1">What’s Inside (PINN)</a></li>
    <li><a href="#-key-concepts-1">Key Concepts (PINN)</a></li>
    <li><a href="#-cases-modeled">Cases Modeled</a></li>
    <li><a href="#preview-sample-code-snippets">Preview (Sample Code)</a></li>
    <li><a href="#technologies-used">Technologies Used</a></li>
    <li><a href="#-output-of-pinn">Output of PINN</a></li>
    <li><a href="#-output-of-methods">Output of Methods</a></li>
    <li><a href="#-citation">Citation</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>



## About
a various numerical methods for solving ode's and machine-learning schemes .
* <a href =  "https://www.canva.com/design/DAGq1cX-lsk/ITrioljMl9J9iHpE72KnxQ/edit"  />Presentation</a>
## 💻 Built Using 
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
* ![Scipy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



---

## 📦 Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/Ibrahim-Abdelqader/ML-Numerical-Diabetes-Glucose-ODE-Modeling.git
cd pinn-glucose-model
```

#### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, use:
```bash
pip install tensorflow numpy matplotlib scipy
```

### 🧪 Running the Notebook

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `PINN.ipynb` in your browser.

3. Run all cells to train the model and visualize glucose-insulin dynamics.

---


# 📘 Numerical Methods for Solving ODEs in Glucose-Insulin Modeling 
This notebook implements and compares **five numerical methods** for solving systems of ordinary differential equations (ODEs) in the context of **glucose-insulin dynamics**. Each method is applied to a system of biological ODEs with time-varying input and conditional behavior.

---
## 🚀 What's Inside?
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Description</th>
      <th>Book</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ISODA</td>
      <td>A stiff ODE solver using adaptive step-size control and automatic method switching (e.g., BDF and Adams). Often used in real-world stiff ODE systems.</td>
      <td>✅</td>
    </tr>
	     <tr>
      <td>Classical 4th-order Runge-Kutta method</td>
      <td>Widely used, fixed-step solver. Computes intermediate slopes (k1–k4) to estimate the solution with good accuracy.</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Newton-Raphson with Backward Euler</td>
      <td>A first-order implicit method solving nonlinear equations at each step using Newton-Raphson. Very stable for stiff problems but requires Jacobian.</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>Trapezoidal</td>
      <td>An implicit method averaging the slope at the current and next time steps. It’s A-stable and more suited for moderately stiff problems.</td>
      <td>❌</td>
    </tr>
 
<tr>
      <td>Midpoint</td>
      <td>A second-order explicit method that estimates midpoint to improve over Euler’s method. More accurate but still not suitable for stiff equations.</td>
      <td>❌</td>
    </tr>
  </tbody>
	
</table>

---

## 📊 Key Concepts

| Concept                 | Description |
|------------------------|-------------|
| **ODE System**         | Models glucose (`G`) and insulin (`I`) based on medical equations and physiological thresholds. |
| **Conditional Logic**  | The model includes threshold behavior (`Gk`, `G0`) affecting both glucose and insulin rates. |
| **Glucose Infusion**   | Modeled as a time-based piecewise function to simulate external glucose input. |

---

## 📂 Methods Breakdown

### 🧮 1. ISODA (via `solve_ivp`)
- Uses `scipy.integrate.solve_ivp` with default (LSODA-like) method.
- Automatically switches between stiff and non-stiff solvers.
- Simple to use, good for reference or real-world deployment.

### ⚙️ 2. Classical Runge-Kutta (4th Order)
```python
for i in range(n):
    k1 = h * f(t[i], y[i], params)
    k2 = h * f(t[i] + h/2, y[i] + k1/2, params)
    k3 = h * f(t[i] + h/2, y[i] + k2/2, params)
    k4 = h * f(t[i] + h, y[i] + k3, params)
    y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
```

### 🔁 3. Trapezoidal Method (Implicit)
- Uses the average of current and next step derivatives.
- Solved using fixed-point iteration:
```python
y_next = y_prev + h/2 * (f(t_prev, y_prev) + f(t_next, y_next))
```

### 🔄 4. Backward Euler with Newton-Raphson
- Very stable for stiff ODEs.
- Iteratively solves:
```python
y_next = y_prev + h * f(t_next, y_next)
```
- Uses numerical Jacobians to perform Newton-Raphson iteration.

### ➕ 5. Midpoint Method (Explicit, 2nd Order)

- A second-order Runge-Kutta method (also called RK2).
- Improves accuracy over Forward Euler by estimating the derivative at the midpoint of the interval.
- More stable than Euler, but not suitable for stiff systems.

```python
for i in range(n):
    k1 = h * f(t[i], y[i], params)
    k2 = h * f(t[i] + h / 2, y[i] + k1 / 2, params)
    y[i+1] = y[i] + k2
```
---


# 🧠 Physics-Informed Neural Network (PINN) for Glucose-Insulin Dynamics

---

## 📋 Description

The notebook simulates and predicts **glucose (G)** and **insulin (I)** behavior using a neural network trained not only on data but also on the governing differential equations of the biological system. It addresses four physiological cases involving glucose infusion and pancreatic sensitivity variation.

---
## 🚀 What’s Inside?

| Section                          | Description                                                                                   |
|----------------------------------|-----------------------------------------------------------------------------------------------|
| `Imports`                        | TensorFlow, NumPy, SciPy, and Matplotlib for modeling and plotting.                          |
| `Parameters & Initial Conditions`| Defines constants like glucose clearance rate, pancreatic sensitivity, and initial states.   |
| `Case Definitions`               | Four experimental cases simulating different biological conditions (infusion & sensitivity). |
| `Glucose Infusion Function`      | Models glucose input over time as a step function.                                           |
| `Smooth Switch Function`         | Implements a sigmoid-based smooth transition to avoid discontinuities.                      |
| `PINN Model Class`               | A deep neural network using `swish` activations to predict glucose and insulin dynamics.     |
| `Loss Function`                  | Combines data loss with residuals from differential equations (`dG/dt`, `dI/dt`).            |
| `Training Loop`                  | Trains the PINN by minimizing the residual-based loss using TensorFlow optimizers.          |
| `Visualization`                  | Plots predicted glucose-insulin curves for all cases.                                        |


This project complements traditional ODE-based simulation of glucose and insulin dynamics by applying a deep learning technique known as a **Physics-Informed Neural Network (PINN)**. 

PINNs embed the system's **physical laws (ODEs)** directly into the neural network’s training process. This allows the network to learn biologically plausible and physically consistent behavior from time-series data, **without requiring large training datasets**.

---

## 📌 Overview
- **Problem domain**: Simulating glucose and insulin concentration in response to a glucose tolerance test.
- **Approach**: Use of a PINN that enforces the underlying ODEs governing glucose-insulin kinetics.
- **Framework**: TensorFlow 2.x + SciPy + Matplotlib
- **Models evaluated**: 4 cases with different infusion levels and pancreatic sensitivity

---

## 🧠 What is a Physics-Informed Neural Network (PINN)?
A Physics-Informed Neural Network (PINN) is a neural network that learns directly from the governing physical laws of the system, typically written as **ODEs or PDEs**. Instead of just predicting outputs from inputs, a PINN is trained so that its outputs satisfy the governing differential equations using **automatic differentiation**.

PINNs embed the system's physical laws (ODEs) directly into the neural network’s training process. This allows the network to learn biologically plausible and physically consistent behavior directly from the governing equations, without requiring any labeled or time-series data. — making them a form of **unsupervised learning constrained by physics**.

### 🔬 Key Benefits of PINNs:
- Learn solutions to differential equations
- Embed known physical laws in the model
- Require little or no labeled data
- Allow real-time or parallel simulations once trained

---

## 🏗️ Neural Network Architecture
- **Input layer**: \( t \in \mathbb{R} \) (time)
- **Hidden layers**: 4 fully connected layers with 128 neurons each
- **Activation function**: `swish` (smooth, non-monotonic, better for gradient flow)
- **Output layer**: \( [G(t), I(t)] \) — predicted glucose and insulin concentrations

---

## 📉 Loss Function
The model is trained to minimize the following composite loss function:

```math
\mathcal{L}_{\text{total}} = \text{MSE}_{\text{ODE}} + 100 \cdot \text{MSE}_{\text{IC}}
```

Where:
```math
( \text{MSE}_{\text{ODE}} = \left( \frac{dG}{dt} - \text{ODE}_G \right)^2 + \left( \frac{dI}{dt} - \text{ODE}_I \right)^2 )
 ```
```math
( \text{MSE}_{\text{IC}} = (G(0) - G_{\text{init}})^2 + (I(0) - I_{\text{init}})^2 )
```
> The ODE residual is the difference between what the ODE says should happen and what the neural network predicts is happening. A perfect solution would have a residual of zero.

- In PINNs, automatic differentiation is used to compute the exact derivative of the neural network's predicted function, not the true derivative of the physical system. This predicted derivative is then compared to the right-hand side of the governing differential equation. The difference (residual) is minimized during training to guide the network toward a solution that obeys the underlying physics.
---

## ⚙️ Training Strategy
The training process used a **two-stage optimization** strategy:

### 1️⃣ Adam Optimizer (Pretraining)
- First-order gradient-based optimizer
- Fast convergence and adaptive learning rates
- 10000 epochs for coarse convergence

### 2️⃣ L-BFGS-B Optimizer (Fine-tuning)
- A second order optimizer (uses curvature information )
- Quasi-Newton second-order optimizer
- Ideal for smooth problems like ODE residual minimization
- Greatly improves precision after Adam

> **Why two optimizers?** Adam moves weights quickly toward a good region. L-BFGS-B fine-tunes that solution to high accuracy using curvature information.

---

## ⏱️ Time Domain Setup
- Domain: \( t \in [0, 12] \) hours
- High resolution during early spike (0–2h), coarser resolution after
- Smooth sigmoid-based switching handles different physiological regimes in the ODE system

---

## 🧪 Initial Conditions
Set to physiological baselines:
- \( G(0) = 81.14 \) mg/dL
- \( I(0) = 5.671 \) mg/dL

---

## 📊 Results Summary
| Case | Description | Max Glucose Error | Max Insulin Error | Training Time (s) | PINN Inference Time (s) |
|------|-------------|-------------------|-------------------|--------------------|--------------------------|
| 1    | Normal, No Infusion | 0.159 | 0.035 | 1776.06 | 0.0346 |
| 2    | Normal, Infusion | 1.813 | 0.305 | 2144.65 | 0.0152 |
| 3    | Reduced Sensitivity | 9.401 | 0.767 | 2219.87 | 0.0164 |
| 4    | Elevated Sensitivity | 1.118 | 0.318 | 2289.87 | 0.0149 |

---

## 💬 Discussion
The PINN approach successfully captured the nonlinear dynamics of the glucose-insulin system across various physiological scenarios.

- PINNs were especially effective in Cases 1, 2, and 4.
- **Case 3** (reduced pancreatic sensitivity) had higher error due to:
  - Weaker feedback (small \( B_b \))
  - Slower glucose decay
  - More sensitive, stiff dynamics over long time spans

> **Why this happens**: In nonlinear and stiff systems, PINNs can struggle with long-term accuracy because they rely solely on minimizing residuals — without corrective data.
> However, **this can be decreased by increasing the number of training epochs** specifically for Case 3, allowing the neural network more time to learn these slow-changing dynamics. The trade-off is **increased training time**, as more epochs demand more computational resources.
### 🔄 Interpretation of ODE Residuals
- PINNs use **automatic differentiation** to compute ``` dG/dt , dI/dt ```
 from the neural network output.
- These derivatives are compared with the RHS of the ODE.
- The difference (residual) is used in training to nudge the network toward physics-consistent solutions.

> Automatic differentiation does not give the "true" physical derivative — it gives the derivative of what the network has learned so far.

---

## ⚡ Performance Benefits of PINNs
While PINNs are more expensive to train initially (30–38 minutes per case), they:
- Are **faster at inference time** (15–35 ms)
- Enable **real-time simulation**
- Support **parameter studies and unseen scenarios** without retraining

---

## ✅ Conclusion
PINNs offer a powerful alternative to traditional solvers in biomedical modeling. By embedding differential equations directly into neural networks, they:
- Learn without labeled data
- Capture physical constraints
- Generalize across cases

This hybrid approach merges the flexibility of deep learning with the rigor of physics, making it highly suitable for applications like **Glucose Tolerance Testing (GTT)** and beyond.


## 📊 Cases Modeled

| Case | Description |
|------|-------------|
| **1** | Normal patient, no infusion |
| **2** | Normal patient, with infusion |
| **3** | Reduced pancreatic sensitivity |
| **4** | Elevated pancreatic sensitivity |

---

<h2 id="preview-sample-code-snippets">🖼️ Preview (Sample Code Snippets)</h2>


### Define Glucose Infusion Function

```python
def glucose_infusion(t, Gt_val):
    return tf.where(t < 0.5, tf.constant(Gt_val, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
```

### PINN Model Architecture

```python
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='swish')
        self.d2 = tf.keras.layers.Dense(128, activation='swish')
        self.d3 = tf.keras.layers.Dense(128, activation='swish')
        self.d4 = tf.keras.layers.Dense(128, activation='swish')
        self.out = tf.keras.layers.Dense(2, activation=None)

    def call(self, t):
        x = self.d1(t)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.out(x)
```

### Loss Function with Gradients

```python
def loss_fn_detailed(model, t, Gt_val, Bb_val):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        pred = model(t)
        G = pred[:, 0:1]
        I = pred[:, 1:2]

    dG_dt = tape.gradient(G, t)
    dI_dt = tape.gradient(I, t)
    del tape
```

---

<h2 id="technologies-used">🛠️ Technologies Used</h2>

- **TensorFlow 2.x** – for defining and training the PINN.
- **SciPy** – for solving ODEs using `solve_ivp` (for ground truth comparison).
- **NumPy & Matplotlib** – for numerical ops and plotting.

---

## 📂 Output of PINN

Plots of glucose and insulin vs. time for each of the 4 cases, comparing PINN predictions to reference ODE solutions.

---
<div name="Screenshots" align="center">
   <img width=60% src="PINN_Screenshots/1.png" alt="logo">
   <hr>
	
   <img width=60% src="PINN_Screenshots/2.png" alt="logo">
  
   <hr>
          <img width=60% src="PINN_Screenshots/3.png" alt="logo">
    <hr>
   <img width=60% src="PINN_Screenshots/4.png" alt="logo">
   <hr>
</div>

---

## 📂 Output of methods

The notebook provides **visual plots** for each method:
- Time-series of **glucose and insulin** concentrations.
- Comparison between methods.

<div name="Screenshots" align="center">
   <img width=60% src="Methods_Screenshots/Screenshot_1.png" alt="logo">
   <hr>
	
   <img width=60% src="Methods_Screenshots/Screenshot_2.png" alt="logo">
  
   <hr>
          <img width=60% src="Methods_Screenshots/Screenshot_3.png" alt="logo">
    <hr>
   <img width=60% src="Methods_Screenshots/Screenshot_4.png" alt="logo">
   <hr>
   <img width=60% src="Methods_Screenshots/Screenshot_5.png" alt="logo">
   <hr>
	
   <img width=60% src="Methods_Screenshots/Screenshot_6.png" alt="logo">
  
   <hr>
          <img width=60% src="Methods_Screenshots/Screenshot_7.png" alt="logo">
    <hr>
   <img width=60% src="Methods_Screenshots/Screenshot_8.png" alt="logo">
   <hr>
      <hr>
   <img width=60% src="Methods_Screenshots/Screenshot_9.png" alt="logo">
   <hr>
      <hr>
   <img width=60% src="Methods_Screenshots/Screenshot_10.png" alt="logo">
   <hr>
</div>


---
## 🧪 Citation

If you use this work in your research or learning, consider citing the original PINN paper:
> M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. JCP, 2019.

---
## Contributors <a name = "contributors"></a>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/hamdy-cufe-eng" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/183446123?s=96&v=4" width="100px;" alt="Hamdy Ahmed"/><br />
        <sub><b>Hamdy Ahmed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Karim-Mohamed-Elsayed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/183163245?v=4" width="100px;" alt="Karim Mohamed"/><br />
        <sub><b>Karim Mohamed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/David-Amir-18" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/183446535?v=4" width="100px;" alt="David Amir"/><br />
        <sub><b>David Amir</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/KareemFareed06" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/205330006?v=4" width="100px;" alt="Kareem Fareed"/><br />
        <sub><b>Kareem Fareed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Jananour00" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/193429413?v=4" width="100px;" alt="Jana Nour"/><br />
        <sub><b>Jana Nour</b></sub>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/MohamedSayed-2005" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/191911134?v=4" width="100px;" alt="Mohamed Sayed"/><br />
        <sub><b>Mohamed Sayed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Jiro75" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/183841405?v=4" width="100px;" alt="Mostafa Hany"/><br />
        <sub><b>Mostafa Hany</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ibrahim-Abdelqader" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/180881453?v=4" width="100px;" alt="Ibrahim Abdelqader"/><br />
        <sub><b>Ibrahim Abdelqader</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SulaimanAlfozan" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/191874168?v=4" width="100px;" alt="Sulaiman"/><br />
        <sub><b>Sulaiman</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/OmegasHyper" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/180775212?v=4" width="100px;" alt="Mohamed Abdelrazek"/><br />
        <sub><b>Mohamed Abdelrazek</b></sub>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/NARDEEN-UX" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/186333337?v=4" width="100px;" alt="Nardeen Ezz"/><br />
        <sub><b>Nardeen Ezz</b></sub>
      </a>
    </td>
  </tr>
</table>
