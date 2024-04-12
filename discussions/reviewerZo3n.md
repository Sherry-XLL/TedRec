Dear Reviewer Zo3n,

Thanks very much for your careful reading and insightful suggestions of our paper! We appreciate the time and efforts you have dedicated to reviewing our work. We list our response to your concern in the following three aspects. If you have further questions, please feel free to let us know. We will continue to try our best to answer for you.

**1. [Q1] Explanation of the Modulation Embeddings in Section 3.1.1.**

The transformation of textual data into the frequency domain can sometimes lead to the amplification of noise. In order to attenuate noise of text, our MoE adapter is based on **parametric whitening**, which incorporates learnable parameters in the whitening transformation for denoised textual representations with better generalizability. 
As for text sequence representation $T$, our implemented modulation embedding in Eq. (4) ($s \in \mathbb{R}^{n \times d'}, n$ is the sequence length and $d'$ is the original dimension size of pre-trained language models) in the text adapter can be interpreted as the **positional modulation for text embeddings within parametric whitening**. That is to say, we set corresponding positional embedding for each position in the text sequence to better capture the temporal semantic features, while traditional fusion strategies do not separately consider the positional encoding for text sequences. The initialization of Modulation Embeddings is the same as other trainable parameters with the default Xavier distribution.

**2. [Q2] Explanation of the performance decrease on variant (w/o IF) for ML-1M dataset.**

Among the five datasets we use, the ML-1M dataset is an extremely dense dataset. At the same time, we obtain the text by concatenating movie titles, release years, and genres, with an **average word count of 5.73 per item text**. The overly rich ID sequences and limited text make the recommendation performance more dependent on temporal modeling of ID sequences, so FEARec using only IDs can achieve suboptimal results in our main table on ML-1M.

As for the fusion method in TedRec, the core component is the Text-ID Mutual Filtering in Section 3.1.2, where the fused representation for a sequence with length $n$ is denoted as $V$, and $V$ consists of the ID sequence embedding $E$, text sequence embedding $T$ and global learnable filter $W$ ($V, E, T, W \in \mathbb{R}^{n \times d}$, gate is ignored): 
$$
V = \mathcal{F}^{-1}(\underbrace{\mathcal{F}(E) \odot \mathcal{F}(T)}_{\text{ID-text convolution}}) + \mathcal{F}^{-1}(\underbrace{\mathcal{F}(E) \odot \mathcal{F}(W)}_{\text{ID-position convolution}}), \\
V = \mathcal{F}^{-1}(\underbrace{\mathcal{F}(E) \odot \mathcal{F}(T + W)}_{\text{convolution between ID and position-aware text}}).
$$
Note that $E$ and $T$ vary from sequence to sequence since each item has a unique ID and text. However, $W$ is the global filter for all sequences, which is independent of items and only related to positions. In terms of $T$ and $W$, they fuse ID sequences from two perspectives: $T$ is personalized text embedding for each sequence to integrate semantic representations, while $W$ is the global positional embedding for all sequences to **capture sequential patterns of sequence representation learning**. In sparse datasets, we rely more on text convolution to achieve semantic fusion. In ML-1M, a dataset with scarce text and dense interaction, the global temporal patterns of items are more important for recommendation. Therefore, removing the ID-position convolution (w/o IF) significantly reduces performance. It is precisely in order to make our method robust for different datasets that we have considered both convolution operations in our design. Each convolution has its own emphasis and contribution varies across different datasets.

**3. [C1] Lack of in-depth analysis on the impact of hyper-parameters on the performance of TedRec.**

Hyper-parameter analysis is an important aspect of effectiveness. Due to space limitations in the uploaded paper, we are sorry not to fully analyze the impact of hyper-parameters on the performance of TedRec. If there is a revised version in the future, we will further supplement discussions with respect to hyper-parameters.

In the following tables, we analyze the impact of hyper-parameters on TedRec from four perspectives: number of experts, dimension size, language model scale and learning rate.

|  | ML-1M |  | OR |  |
|:---:|:---:|:---:|:---:|:---:|
| Expert num | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 |
| 1 | 0.2523 | 0.1370 | 0.2172 | 0.1298 |
| 2 | 0.2533 | 0.1394 | 0.2211 | 0.1295 |
| 3 | 0.2601 | 0.1409 | 0.2219 | 0.1284 |
| 4 | 0.2560 | 0.1422 | 0.2215 | 0.1301 |
| 5 | 0.2586 | 0.1435 | 0.2196 | 0.1276 |
| 6 | 0.2540 | 0.1398 | 0.2193 | 0.1301 |
| 7 | 0.2528 | 0.1415 | 0.2222 | 0.1305 |
| 8 | **0.2623** | **0.1445** | **0.2234** | **0.1316** |
| 9 | 0.2543 | 0.1404 | 0.2180 | 0.1285 |
| 10 | 0.2548 | 0.1386 | 0.2189 | 0.1296 |

|  | ML-1M |  | OR |  |
|:---:|:---:|:---:|:---:|:---:|
| Dim size | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 |
| 64 | 0.2159 | 0.1071 | 0.1849 | 0.1016 |
| 128 | 0.2368 | 0.1300 | 0.2067 | 0.1191 |
| 256 | 0.2533 | 0.1411 | 0.2197 | 0.1293 |
| 300 | **0.2623** | **0.1445** | **0.2234** | **0.1316** |
| 512 | 0.2522 | 0.1396 | 0.2132 | 0.1225 |

|  | ML-1M |  | OR |  |
|:---:|:---:|:---:|:---:|:---:|
| PLM size | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 |
| BERT tiny | 0.2550 | 0.1399 | 0.2183 | 0.1283 |
| BERT mini | 0.2545 | 0.1403 | 0.2203 | 0.1291 |
| BERT small | 0.2584 | 0.1414 | 0.2206 | 0.1296 |
| BERT medium | 0.2526 | 0.1385 | 0.2187 | 0.1276 |
| BERT base | **0.2623** | **0.1445** | **0.2234** | **0.1316** |
| BERT large | 0.2583 | 0.1419 | 0.2222 | 0.1303 |

|  | ML-1M |  | OR |  |
|:---:|:---:|:---:|:---:|:---:|
| lr | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 |
| 0.0003 | 0.2609 | 0.1442 | **0.2277** | **0.1355** |
| 0.0005 | 0.2584 | 0.1434 | 0.2230 | 0.1326 |
| 0.001 | **0.2623** | **0.1445** | 0.2234 | 0.1316 |
| 0.003 | 0.2260 | 0.1160 | 0.2179 | 0.1266 |
| 0.005 | 0.1727 | 0.0872 | 0.0455 | 0.0227 |


Thanks again for your valuable comments! If you have further questions about our paper, looking forward to your active discussions! 

Best,

Authors of Submission 707