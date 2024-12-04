# Efficient Deep Learning Course

This repository is designed to complement the **Efficient Deep Learning** course, which addresses the challenges posed by modern neural networks. These networks often demand substantial memory and computational resources, making their deployment on mobile and edge devices challenging. Additionally, the increasing scale and complexity of these models make training resource-intensive, creating bottlenecks that hinder the advancement of AI applications.

## Course Overview

The course is structured into two primary sections:

### 1. Improving Inference Efficiency
In this part, students learn how to enhance the efficiency of neural network inference by:
- **Evaluating Neural Network Effectiveness:** Understanding performance trade-offs.
- **Applying Compression Techniques:** Techniques such as:
  - **Pruning:** Removing redundant network parameters.
  - **Tensor Factorization:** Decomposing tensors for computational savings.
  - **Quantization:** Reducing precision to create smaller, faster models without compromising accuracy.

### 2. Optimizing the Training Process
This section focuses on techniques to address the challenges of scaling and training modern AI models, including:
- **Memory-Saving Methods:**
  - **Re-materialization (Activation Checkpointing):** Reducing memory usage during backpropagation.
  - **Offloading:** Moving intermediate computations to external storage when needed.
- **Parallelism Strategies:**
  - **Data Parallelism:** Distributing data across multiple GPUs or nodes.
  - **Tensor Parallelism:** Splitting tensor operations across devices.
  - **Model Parallelism:** Dividing model components for efficient training.
  - **Pipeline Parallelism:** Staggered execution of model layers to reduce idle time.

## Profiling Neural Networks
Throughout the course, a strong emphasis is placed on profiling neural networks to:
- Identify performance bottlenecks.
- Develop practical solutions for optimizing both inference and training.

By the end of this course, students will have the skills to:
- Optimize the performance of neural networks.
- Deploy and train advanced AI models in real-world scenarios efficiently.

### Contributions
Contributions to improve this repository and course materials are welcome. Please follow the contribution guidelines provided.

---

Explore, experiment, and build efficient AI solutions!
