# AFERSM-Net: Auxiliary Feature Extraction based Residual Shrinkage Multi-tasking Network 

To develop a method to address the multipath effect noise in WiFi-based gesture recognition and location classification, and mitigate the impact of amplitude changes caused by location variation on gesture feature extraction and recognition.
Auxiliary Feature Extraction based Residual Shrinkage Multi-tasking Network (AFERSM-Net) is proposed for gesture recognition and position classification of one-dimensional multivariate time series. AFERSM-Net is a hybrid architecture that combines CNN for spatial feature extraction and LSTM networks for capturing temporal dependencies. Firstly, a reasonable threshold is set adaptively by the shrinkage module to dynamically identify and eliminate the transformed environmental noise. Secondly, the feature extraction module is used to focus on and extract location-independent gesture features to reduce the influence of location-independent features. Finally, the gesture features extracted by the feature extraction module are fused with the shared features of the residual shrinkage multi-tasking network as an aid. Its module fusion is mainly used to improve the accuracy of gesture recognition and solve the problem of insufficient model generalization ability.
The AFERSM-Net model was evaluated on the dual-labeled gesture and location dataset, achieving 97.84% accuracy for gesture recognition and 98.92% accuracy for location classification. 

![](https://github.com/Purecade-czx/AFERSM-Net/blob/main/Fig2.png)

## Usage

1.Please download [data](https://drive.google.com/open?id=1SCxUHbl6rNWM3kT0c-D4s_kyAero9_-o
), and decompress it at the root folder of this repository.

> Activity Label: 0. hand up;  1. hand down; 2. hand left; 3. hand right; 4. hand circle; 5. hand cross.
> Location Label: 0, 1, 2, ..., 15

2.Please download [pre-trained weights](https://drive.google.com/open?id=1UT61Gs746yijxiKvLyP0wHIPxi9MYz0Y), and decompress it at the root folder of this repository.

3.Please download the complete GitHub project.

## Environment

a.Please run **environment. yml** while ensuring that the software system has Conda.

```python
conda env create -f environment.yml
```

b.Please store the dataset folder and program files in the same directory.

```
.
├── data
├── README.md
├── models
├── sae_train.py
├── environment.yml

```



## Run

Run sae_train.py
