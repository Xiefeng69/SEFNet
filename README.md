[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=Xiefeng69.SEFNet
[repo-url]: https://github.com/Xiefeng69/SEFNet

# SEFNet

[![visitors][visitors-img]][repo-url]

[SEKE2022] This is a PyTorch implementation of the paper: Inter- and Intra-**S**eries **E**mbeddings **F**usion **Net**work for Epidemiological Forecasting (SEFNet) \[[paper](http://ksiresearch.org/seke/seke22paper/paper109.pdf)\].

## Introduction
The accurate forecasting of infectious epidemic diseases is the key to effective control of the epidemic situation in a region. Most existing methods ignore potential dynamic dependencies between regions or the importance of temporal dependencies and inter-dependencies between regions for prediction. In this paper, we propose an Inter- and Intra-Series Embeddings Fusion Network (SEFNet) to improve epidemic prediction performance. SEFNet consists of two parallel modules, named Inter-Series Embedding Module and Intra-Series Embedding Module. In Inter-Series Embedding Module, a multi-scale unified convolution component called Region-Aware Convolution is proposed, which cooperates with self-attention to capture dynamic dependencies between time series obtained from multiple regions. The Intra-Series Embedding Module uses Long Short-Term Memory to capture temporal relationships within each time series. Subsequently, we learn the influence degree of two embeddings and fuse them with the parametric-matrix fusion method. To further improve the robustness, SEFNet also integrates a traditional autoregressive component in parallel with nonlinear neural networks. Experiments on four real-world epidemic-related datasets show SEFNet is effective and outperforms state-of-the-art baselines.

## Dataset
**In the folder: /data**

:gift_heart: Note: the Influenza-related datasets are released by [Cola-GNN](https://github.com/amy-deng/colagnn) and the COVID-related data is publicly avaliable at [JHU-CSSE](https://github.com/CSSEGISandData/COVID-19).

`/data/us-state-label.xlsx` is the corresponding index file to the `/data/state360.txt`, noting that in `/data/state360.txt`, FL (idx=8) is removed due to missing value.

|  Data set   | Size  |  Min  | Max  | Mean  | SD  |  Granularity
|  ----  | ----  |  ----  | ----  |  ----  | ----  | ----  |
| US-Regions  |  10×785 | 0 |  16526 | 1009 | 1351 | weekly
| US-States  |  49×360 | 0 | 9716 | 223 | 428 | weekly
| Japan-Prefectures  |  47×348 | 0 |  26635 | 655 | 1711 | weekly
| Canada-Covid  |  13×717 | 0 | 127199 | 3082 | 8473 | daily

## Quick Start

All programs are implemented using Python 3.8.5 and PyTorch 1.9.1 with CUDA 11.1 (1.9.1 cu111) in an Ubuntu server with an Nvidia Tesla K80 GPU.

Dependency can be installed using the following command:
```shell
pip install -r requirements.txt
```

+ For dataset *US-Regions*:
```shell
python src/train.py --lr 0.01 --hw 20 --hidP 3 --hidR 32 --hidA 32 --k 8 --data region785 --horizon 5 --gpu 0
```
+ For dataset *US-States*:
```shell
python src/train.py --epochs 2000 --lr 0.01 --hw 20 --hidR 64 --hidA 64 --k 8 --data state360 --horizon 5 --gpu 0
```
+ For dataset *Canada-Covid*
```shell
python src/train.py --epochs 2000 --lr 0.01 --data canada-covid --horizon 3 --gpu 0 --model model --hidA 96 --hidR 96 --hidP 1
```

## SEFNet's Parameters
|  Parameter | Explanation |
|  ---- | ----  |
| k | Number of kernels of each scale convolution in RAConv |
| n_layer | Number of LSTM layers  |
| hidR | The hidden dimension of LSTM |
| hidA | The hidden dimension of Attention Layer |
| hidP | The ouput dimension of Adaptive Pooling |
| lr | Learning rate |
| hw | Look-back window of AutoRegressive component |

## Citation
```
@inproceedings{xie2022sefnet,
  title={Inter- and Intra-Series Embeddings Fusion Network for Epidemiological Forecasting},
  author={Xie, Feng and Zhang, Zhong and Zhao, Xuechen and Zhou, Bin and Yu, Songtan},
  year={2022},
  booktitle={The 34th International Conference on Software Engineering & Knowledge Engineering},
  organization={KSIResearch}
}
```