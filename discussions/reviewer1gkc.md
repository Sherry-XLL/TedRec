Dear Reviewer 1gkc,

Thanks very much for your careful reading and insightful suggestions of our paper! We appreciate the time and efforts you have dedicated to reviewing our work. We list our response to your concern in the following three aspects. If you have further questions, please feel free to let us know. We will continue to try our best to answer for you.

**1. [W1] Efficiency comparison of TedRec.**

In addition to training efficiency, the efficiency of model prediction and inference is also an important aspect of effectiveness. Due to space limitations in the uploaded paper and similar trends in training and inference efficiency, we are sorry not to fully demonstrate the inference efficiency of our method. If there is a revised version in the future, we will further supplement discussions with respect to inference efficiency.

As shown in Table 1, we conduct experiments on NVIDIA A100 Tensor Core GPU with CUDA Version 11.4. In order to ensure fairness in efficiency comparison, we only train one model on the GPU at a time (to avoid the problem of parallel training affecting model efficiency), and then report the average training and inference time for each epoch, measured in seconds. It can be seen from Table 1 that our proposed TedRec has significantly superior recommendation performance, and **the training and inference time is also comparable to other fusion methods**. 

Table 1. Overall efficiency and performance comparison.

| Dataset | Metric | SASRec | FMLPRec | FEARec | SASRecF | FDSA | DIF-SR | UniSRec | DLFS-Rec | SMLP4Rec | Ours |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ML-1M | Training | **49.59** | 59.82 | 644.84 | 59.20 | 100.84 | 85.31 | 102.51 | 848.17 | 118.58 | 109.97 |
|  | Inference | **0.11** | 0.14 | 0.74 | 0.13 | 0.20 | 0.17 | 0.21 | 3.36 | 0.23 | 0.22 |
|  | Recall@10 | 0.2300 | 0.2363 | 0.2407 | 0.2356 | 0.2286 | 0.2313 | 0.2257 | 0.2179 | 0.2157 | **0.2623** |
| OR | Training | **24.15** | 29.33 | 271.91 | 28.94 | 48.63 | 41.38 | 49.82 | 389.57 | 57.48 | 52.83 |
|  | Inference | **0.32** | 0.37 | 0.50 | 0.37 | 0.76 | 0.48 | 0.56 | 9.31 | 0.90 | 0.58 |
|  | Recall@10 | 0.1545 | 0.1492 | 0.1532 | 0.1479 | 0.1491 | 0.1522 | 0.1526 | 0.1634 | 0.1515 | **0.2234** |
| Office | Training | **25.99** | 31.04 | 315.45 | 29.59 | 50.76 | 43.43 | 55.50 | 3013.66 | 59.49 | 53.06 |
|  | Inference | 5.37 | 7.40 | 7.52 | **4.67** | 6.30 | 5.83 | 5.95 | 302.78 | 6.18 | 6.32 |
|  | Recall@10 | 0.1061 | 0.1138 | 0.1172 | 0.1081 | 0.1111 | 0.1162 | 0.1233 | 0.1131 | 0.1183 | **0.1356** |
| Food | Training | **44.89** | 52.27 | 538.39 | 49.23 | 84.33 | 71.58 | 91.13 | 7717.44 | 93.66 | 85.12 |
|  | Inference | **6.21** | 7.16 | 10.85 | 6.32 | 11.66 | 9.33 | 9.20 | 631.82 | 8.75 | 9.79 |
|  | Recall@10 | 0.1069 | 0.1133 | 0.1192 | 0.1084 | 0.1100 | 0.1144 | 0.1259 | 0.1146 | 0.1160 | **0.1327** |
| Movies | Training | **161.27** | 183.48 | 1960.75 | 181.08 | 289.50 | 268.38 | 346.41 | 39224.13 | 326.44 | 303.93 |
|  | Inference | **16.87** | 16.35 | 17.72 | 21.44 | 20.92 | 29.50 | 20.40 | 1933.90 | 19.89 | 19.20 |
|  | Recall@10 | 0.1453 | 0.1477 | 0.1496 | 0.1425 | 0.1446 | 0.1440 | 0.1493 | 0.1504 | 0.1425 | **0.1611** |

**2. [W1]  The impact of data scale on efficiency.**

We further explore the impact of data scale on efficiency and effectiveness with different sizes of MovieLens datasets. As shown in Table 2, it can be seen that data size is the most direct factor affecting efficiency.

Table 2. Efficiency comparison of different data scales.

|  | ML-1M |  |  | ML-10M |  |  | ML-20M |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Method | Recall@10 | Training | Inference | Recall@10 | Training | Inference | Recall@10 | Training | Inference |
| SASRec | 0.2300 | **49.59** | **0.11** | 0.1905 | **872.16** | **4.37** | 0.1752 | **1883.63** | **7.83** |
| SASRecF | 0.2356 | 59.20 | 0.13 | 0.1871 | 895.56 | 4.53 | 0.1759 | 2149.95 | 8.08 |
| UniSRec | 0.2257 | 102.51 | 0.21 | 0.1748 | 1126.46 | 4.62 | 0.1690 | 2294.38 | 8.56 |
| TedRec | **0.2623** | 109.97 | 0.22 | **0.2157** | 1151.06 | 4.82 | **0.2048** | 2266.31 | 8.28 |

**3. [W1]  The impact of hyper-parameters on efficiency.**

As shown in Table 3 and Table 4, we can see that the number of experts in multi-expert text adapter has significant impact on training time with marginal impact on inference time. As for the dimension size, both training and inference efficiency are affected by it.

Table 3. The impact of expert num on efficiency of TedRec.

|  | ML-1M |  |  | OR |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Expert Num | Recall@10 | Training | Inference | Recall@10 | Training | Inference |
| 1 | 0.2523 | **73.73** | 0.23 | 0.2172 | **35.14** | 0.57 |
| 2 | 0.2533 | 78.93 | 0.22 | 0.2211 | 38.09 | 0.59 |
| 3 | 0.2601 | 85.16 | **0.21** | 0.2219 | 41.43 | 0.58 |
| 4 | 0.2560 | 90.25 | 0.22 | 0.2215 | 44.12 | 0.61 |
| 5 | 0.2586 | 96.42 | 0.22 | 0.2196 | 47.41 | 0.62 |
| 6 | 0.2540 | 103.63 | 0.23 | 0.2193 | 49.82 | 0.62 |
| 7 | 0.2528 | 109.28 | 0.22 | 0.2222 | 53.39 | 0.56 |
| 8 | **0.2623** | 109.97 | 0.22 | **0.2234** | 52.83 | 0.58 |
| 9 | 0.2543 | 120.32 | 0.24 | 0.2180 | 58.19 | **0.55** |
| 10 | 0.2548 | 125.97 | 0.24 | 0.2189 | 60.88 | 0.59 |

Table 4. The impact of dimension size on efficiency.

|  | ML-1M |  |  | OR |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dim Size | Recall@10 | Training | Inference | Recall@10 | Training | Inference |
| 64 | 0.2159 | **52.95** | **0.17** | 0.1849 | **43.22** | **0.34** |
| 128 | 0.2368 | 62.87 | 0.18 | 0.2067 | 44.06 | 0.41 |
| 256 | 0.2533 | 102.04 | 0.20 | 0.2197 | 47.82 | 0.57 |
| 300 | **0.2623** | 109.97 | 0.22 | **0.2234** | 52.83 | 0.58 |
| 512 | 0.2522 | 186.96 | 0.39 | 0.2132 | 86.65 | 1.02 |

Thanks again for your valuable comments! If you have further questions about our paper, looking forward to your active discussions! 

Best,

Authors of Submission 707