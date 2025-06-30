## Environment

```
GPU RTX 3090(24GB) * 1
python==3.7.16
pytorch=1.8.0
pip=22.3.1
```



## Usage

```python
# Install dependencies
conda env create -f environment.yml / 
pip install -r requirements.txt

# Train and test Proto-CSNet
python mix_test.py


'''
Baseline Comparisons
'''
# Siamese Network+CNN
python baseline.py

# Siamese Network+ResNet12
python siam_resnet.py

# ProtoNet+CNN
python protonet_baseline.py

# ProtoNet+ResNet12
python protonet_resnet.py

```



## Proto-CSNet Introduction

### Overview
With the emergence of the 5.5G era, **Integrated Sensing and Communication (ISAC)** has demonstrated powerful high-precision sensing capabilities in various application domains, including wireless communications, intelligent transportation, and smart homes. In the field of human activity recognition, researchers are actively exploring the feasibility and convenience of using **Channel State Information (CSI)** extracted from Wi-Fi signals to detect and classify human actions.

However, existing work often relies on complex and heavyweight neural network architectures to process large volumes of CSI data in pursuit of high recognition accuracy. This creates significant challenges in time-sensitive scenarios, where researchers must deal with both **limited training samples** and the demand for **efficient and timely responses**.

To address these issues, we propose **Proto-CSNet**, a **prototypical network model** enhanced with **CNN and self-attention mechanisms**. Our contributions are threefold:
1. We first apply **time-frequency transformations and filtering** to convert the CSI data from the time domain to the frequency domain. Outliers and noise are removed to improve the ability to distinguish between similar activities.
2. We then build a **customized feature extraction module** based on the ACmix model, preserving the local feature extraction power of CNNs and the global modeling strength of self-attention mechanisms.
3. Finally, we integrate this module with a CNN backbone and embed it within a **meta-learning framework** using prototypical networks to support few-shot learning tasks.

Extensive experiments validate the effectiveness of our method. We collect a CSI dataset covering **11 types of human activities** in **two real-world environments** and conduct thorough evaluations. The results show that:
- The inference accuracy exceeds **95%**
- The model achieves significantly faster **training and inference**
- **Proto-CSNet** consistently outperforms existing CSI-based activity recognition methods


### Model Structure：

![image-20250627205545775](README.assets/image-20250627205545775.png)



## Paper Link：

[Proto-CSNet:](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11036372)

```latex
@INPROCEEDINGS{11036372,
  author={Hu, Siyu and Liu, Jiqiang and Zhang, Chenxin and Zhu, Xiaoqiang and Li, Lingkun},
  booktitle={2024 20th International Conference on Mobility, Sensing and Networking (MSN)}, 
  title={Proto-CSNet: A Prototype Network Model Integrating CNN and Self-Attention for Enhanced Human Activity Recognition}, 
  year={2024},
  pages={48-56},
  doi={10.1109/MSN63567.2024.00018}}

```
