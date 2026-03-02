# SynRTP

To ensure reproducibility and a rigorous fair comparison, all experiments are conducted on a unified hardware platform with a single Tesla V100 GPU (16 GB). SynRTP is implemented in PyTorch. For all baseline models, we adopt a standardized evaluation protocol to avoid implementation bias:

**(1) Standardized benchmark configurations**

* **[`LaDe`](https://huggingface.co/datasets/Cainiao-AI/LaDe) benchmark baselines.** Most baselines (including DeepRoute, Graph2Route, etc.) and the datasets used in this paper are taken from the open-source LaDe benchmark repository. To make our results directly comparable with community standards, we strictly use the official implementations and their default optimal hyperparameter settings provided in LaDe.  
* **Independent baselines.** For baselines not included in LaDe (e.g., DutyTTE and MRGRP), we use their official open-source implementations and adopt the default optimal hyperparameter combinations recommended by the original authors. This strategy ensures that every baseline is evaluated close to its intended peak performance, avoiding bias from subjective re-tuning.

**(2) Strict fairness control**

Beyond model configurations, we enforce a unified training protocol across all methods so that no model receives an unfair advantage.  
- **Input consistency.** All models use exactly the same set of input features (spatial coordinates, temporal timestamps, and courier profiles). No baseline is handicapped by missing features, and no model has access to additional information unavailable to others.  
- **Termination criterion.** To prevent over-training or under-training biases, we apply a consistent early-stopping mechanism to all models: training stops if the validation metric (KRC) does not improve for 11 consecutive epochs.

**(3) SynRTP settings**

For SynRTP, hyperparameters are selected based on validation performance: the hidden dimension is set to $d_h = 32$, the Graphormer encoder has 3 layers with 4 attention heads, and the RAPO group sampling size is $G = 16$. We train the model in a two-stage scheme using the Adam optimizer with a learning rate of $1 \times 10^{-4}$.



## Dataset Description

We evaluate SynRTP on four large-scale real-world instant-delivery datasets from the **LaDe** benchmark released by Cainiao Network. In our formulation, couriers are treated as workers and delivery tasks are treated as graph nodes. The four datasets cover diverse cities and logistics environments, including grid-like urban structures (Logistics-SH, Logistics-HZ), mountainous and spatially fragmented regions (Logistics-CQ), and a coastal-belt topology (Logistics-YT). This diversity provides a practical testbed for evaluating both route generation and time prediction under heterogeneous spatial layouts and demand patterns. All datasets are anonymized (e.g., identifier hashing and coordinate offsetting) for privacy protection.


## Data Generation for Model Training

Install environment dependencies using the following command:

```shell
pip install -r requirements.txt
```

After downloading the original datasets, please use the following command to generate the data required for model training:
```shell
bash DataPipeline.sh
```

To facilitate verification of the correctness of the model code, we provide a very small dataset of Logistics-YT, extracting a batch size of 8 from each of the original data training set, validation set and test set (the default batch size of the model dataset is 64).


## Training SynRTP Model


Taking the Logistics-YT dataset as an example. Run the following command to train the SynRTP. 

```shell
python run.py --dataset yt_dataset
```




## Baseline Reproduction

Taking the Logistics-YT dataset as an example. Use the following commands to reproduce baseline models:
```shell
# Time-Greedy
python baselines/LaDe/route_prediction/run.py --model Time-Greedy --dataset yt_dataset

# Distance-Greedy
python baselines/LaDe/route_prediction/run.py --model Distance-Greedy --dataset yt_dataset

# Osquare
python baselines/LaDe/route_prediction/run.py --model Osquare --dataset yt_dataset

# DeepRoute
python baselines/LaDe/route_prediction/run.py --model DeepRoute --dataset yt_dataset

# Graph2Route
python baselines/LaDe/route_prediction/run.py --model Graph2Route --dataset yt_dataset

# DRL4Route
python baselines/LaDe/route_prediction/run.py --model DRL4Route --dataset yt_dataset

# Static-ETA
python baselines/LaDe/time_prediction/run.py --model Static-ETA --dataset yt_dataset

# KNN-MultiETA
python baselines/LaDe/time_prediction/run.py --model KNN-MultiETA --dataset yt_dataset

# XGB-MultiETA
python baselines/LaDe/time_prediction/run.py --model XGB-MultiETA --dataset yt_dataset

# DeepETA
python baselines/LaDe/time_prediction/run.py --model DeepETA --dataset yt_dataset

# DutyTTE
python baselines/DutyTTE/main.py --dataset_name yt_dataset

# RankETPA
python baselines/LaDe/time_prediction/run.py --model RankETPA --dataset yt_dataset

# M2G4RTP
python baselines/LaDe/route_prediction/run.py --model M2G4RTP --dataset yt_dataset

# MRGRP
python baselines/MRGRP/run.py --dataset_name yt_dataset

```


