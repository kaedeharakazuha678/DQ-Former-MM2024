# DQ-Former: Querying Transformer with Dynamic Modality Priority for Cogntive-aligned Multimodal Emotion Recognition in Conversations

This repository contains the code for the paper "DQ-Former: Querying Transformer with Dynamic Modality Priority for Cogntive-aligned Multimodal Emotion Recognition in Conversations".

## 1. Requirements

please refer to `envs/enviroment_abc.yml` see the detailed requirements.

## 2.Preparation

You need to download the datasets and preprocess them before running the code. The data preprocessing steps are provided in the `preprocess` folder.

### 2.1 Raw data download

The data is available at [IEMOCAP](https://sail.usc.edu/iemocap/), and [MELD](https://github.com/declare-lab/MELD.git).

### 2.2 Data preprocessing

Replace the path in `preprocess` with our own data_path. Please refer to the `preprocess` folder for the data preprocessing steps.

### 2.3 Pre-trained unimodel encoders

You need to prepare the pre-trained unimodel encoders, and add the model path to the `scripts/config/config.sh` file.

Please see the needed models in `scripts/config/config.sh`, and download them from huggingface or modelscope.

## 3.Training

### Supported models

- DQ-Former
- Mult
- Self-attention
- MBT
- more models see `my_models`, you can design your own model here.

To train the model, you can run the following command:

```bash
bash scripts/run_train.sh
```

The settings for the training are provided in the `scripts/train/train.sh` file.

## Evaluation

To evaluate the model, you can run the following command:

```bash
bash scripts/run_eval.sh
```

The settings for the evaluation are provided in the `scripts/eval/eval.sh` file.

## Configs

The arguments are designed in `utils/args_utils.py`

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@inproceedings{10.1145/3664647.3681599,
author = {Jing, Ye and Zhao, Xinpei},
title = {DQ-Former: Querying Transformer with Dynamic Modality Priority for Cognitive-aligned Multimodal Emotion Recognition in Conversation},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681599},
doi = {10.1145/3664647.3681599},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {4795–4804},
numpages = {10},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

## Contact

If you have any questions, please feel free to contact me via email yejing2022@ia.ac.cn
