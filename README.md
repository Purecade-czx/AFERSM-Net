# AFERSM-Net: Auxiliary Feature Extraction based Residual Shrinkage Multi-tasking Network 

To develop a method to address the multipath effect noise in WiFi-based gesture recognition and location classification, and mitigate the impact of amplitude changes caused by location variation on gesture feature extraction and recognition.The AFERSM-Net model was evaluated on the dual-labeled gesture and location dataset, achieving 97.84% accuracy for gesture recognition and 98.92% accuracy for location classification. The model outperforms existing network frameworks, based on evaluation metrics such as confusion matrix, precision, recall, specificity, and F1 scores. The proposed network also demonstrates superior performance in handling both tasks when compared to other algorithms, showing its suitability for the CSI dataset.AFERSM-Net demonstrates exceptional performance in both gesture recognition and location classification tasks, outperforming other existing network frameworks. Its high accuracy and robustness make it a promising approach for practical applications in WiFi-based gesture recognition and indoor location classification.

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
