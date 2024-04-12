Dear Reviewer 1Snz,

Thanks very much for your careful reading and insightful suggestions of our paper! We appreciate the time and efforts you have dedicated to reviewing our work. We list our response to your concern in the following four aspects. If you have further questions, please feel free to let us know. We will continue to try our best to answer for you.

**1. [Q1] The effectiveness of the mixture-of-experts (MoE) in improving the discriminability of textual representations.**

To validate the effectiveness our proposed MoE-enhanced text adapter for discriminable textual representations, we perform PCA dimensionality reduction and KDE kernel density estimation on the textual representations of ML-1M and OR datasets before and after MoE, respectively. As illustrated in Figure 1, we can see that our MoE-enhanced text adapter increases the distinguishability of text embedding and provides **smooth anisotropic semantic space** for general texts since representations after MoE are more uniformly distributed within the same coordinate system. 

![](../asset/text_dis.png)

**2. [Q2] The impact of varying the number of experts.**

As for the configuration of MoE, we follow settings in UniSRec and set the number of expert as 8 for all datasets in our experiments. We also conduct experiments to analyze the impact of the expert on TedRec. As shown in Table 1, we can see that the appropriate number of experts is the key to achieving optimal results, and when the number of experts is 8, better results can be achieved on both datasets.

Table 1. Impact of the number of experts on TedRec.

|  | ML-1M |  | OR |  |
|:---:|:---:|:---:|:---:|:---:|
| Expert Num | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 |
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

**3. [Q3] The relationship between data scale and the efficacy of the sequence-level representation fusion.**

We first explore the performance of TedRec and three baselines on three sizes of MovieLens datasets. As shown in Table 2, it can be seen that the performance improvement of TedRec compared to the runner-up model increases with the increase of interactions.

Table 2. Performance of TedRec with different ML datasets.

|  | ML-1M |  | ML-10M |  | ML-20M |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Method | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 | Recall@10 | NDCG@10 |
| SASRec | 0.2300 | 0.1146 | *0.1905* | *0.0954* | 0.1752 | 0.0876 |
| SASRecF | *0.2356* | *0.1249* | 0.1871 | 0.0950 | *0.1759* | *0.0883* |
| UniSRec | 0.2257 | 0.1140 | 0.1748 | 0.0853 | 0.1690 | 0.0830 |
| TedRec | **0.2623** | **0.1445** | **0.2157** | **0.1154** | **0.2048** | **0.1092** |
| Impr. | 11.33% | 15.69% | 13.23% | 20.96% | 16.43% | 23.67% |

We then explore the performance of TedRec with different scales of training data on the ML-1M dataset. We randomly select 20%, 40%, 60%, and 80% of the interaction data from the original dataset as new training data and report the model results. From Table 3, it can be seen that as the data scale of training data increases, the improvement of TedRec compared to the runner-up model also shows an upward trend.

Table 3. Performance of TedRec (Recall@10) with different scales of training data on the ML-1M dataset. 

| Scale | 20% | 40% | 60% | 80% | 100% |
|:---:|:---:|:---:|:---:|:---:|:---:|
| SASRec | 0.1955 | _0.2187_ | 0.2265 | 0.2227 | 0.2300 |
| SASRecF | 0.1987 | 0.2149 | _0.2295_ | _0.2306_ | _0.2356_ |
| UniSRec | _0.2055_ | 0.2147 | 0.2275 | 0.2242 | 0.2257 |
| TedRec | **0.2084** | **0.2328** | **0.2379** | **0.2546** | **0.2623** |
| Impr. | 1.41% | 6.45% | 3.66% | 10.41% | 11.33% |

**4. The novelty and rationale of TedRec.**

The core component of TedRec is the Text-ID Mutual Filtering.
Unlike prior work (e.g., FMLPRec), we are the **first** to conduct the fusion of *text and IDs* in the frequency domain, which could also provide new insights and performance enhancements for integrating other modalities such as image data.
Specifically, TedRec has two key properties: 1. Contextual invariance. 2. Disentangled contextual integration.
With these superior properties, TedRec could significantly enhance the **generalization** and **scaling** ability of backbone models, which largely improves the sequential modeling performance in modality fusion and large-scale scenarios.

For better understanding, we elaborate on the theoretical properties of TedRec as follows:

**(1) Contextual invariance.**

Previous frequency-based methods such as FMLPRec mainly perform convolution through IDs and filter weights. They fail to capture contextual information of sub-sequences, since the same sub-sequence can have rather different representations at different positions. 
In contrast, the binding effect of IDs and text in TedRec can effectively overcome this issue, enabling it to more efficiently learn user behavior at the sequence level.

For example, in FMLPRec and TedRec, the convolution results of sequence $[a,b,c]$ at different positions are shown as:

Table 4. Example of FMLPRec's contextual variance.

| Position | 1 | 2 | 3 | ... | 61 | 62 | 63 |
|-----|-----|:-----:|-----|-----|-----|:-----:|-----|
| ID | $a$ | $b$ | $c$ |... |a | b | c |
| Conv | ↘ | ↓ | ↙ | - |↘ | ↓ | ↙|
| Weight | $w_1$ | $w_2$ | $w_3$ | ... | $w_{61}$ | $w_{62}$ | $w_{63}$ |
| Fused in FMLPRec | |$aw_3 + bw_2+cw_1$|  | $\neq$ |  | $aw_{63} + bw_{62}+cw_{61}$ |

Table 5. Example of TedRec's contextual invariance.

| Position | 1 | 2 | 3 | ... | 61 | 62 | 63 |
|-----|-----|:-----:|-----|-----|-----|:-----:|-----|
| ID | $a$ | $b$ | $c$ |... |a | b | c |
| Conv | ↘ | ↓ | ↙ | - |↘ | ↓ | ↙|
| Text | $t_a$ | $t_b$ | $t_c$ | ... | $t_a$ | $t_b$ | $t_c$ |
| Fused in TedRec | |$at_c + bt_b+ct_a$|  | $=$ |  | $at_c + bt_b+ct_a$ |

We can see that in FMLPRec, the representation of the sequence [a, b, c] at different positions is significantly different.
Whereas in text-ID fusion of TedRec, the sequential representations are solely based on contextual information and are independent of positions.
This provides a new way to improve the model's generalization ability by **reusing the same context/sub-sequence from different users' sequences**.

**(2) Disentangled contextual integration.**

As proved in our paper (LEMMA 3.2), the text-ID fused sequence representation at the $j$-th position $v_j$ can be modeled by combining both the past and future information of the sequence, i.e., $v_j = S(0, j) + S(j+1, n-1)$, where $S(m, n)$ is the text-ID sequential convolution between the $m$-th and $n$-th position. This property can improve the **generalization capabilities of the sequence-level fusion by reusing learned patterns**.
For example, we consider the following samples:

Table 6. Example of TedRec's disentangled contextual integration.

| Position | 1 | 2 | 3 | ... | 60|61 | 62 | 63 |
|-----|-----|:-----:|:-----:|-----|:-----:|-----|:-----:|-----|
| Seq 1 | $a$ | $b$ | $c$ |... | k| d | e | f |
| Fused Rep | - | - | $S_1(1,3) + S_1(4,63)$ |-| - | - | - | - |
| Seq 2 | $m$ | $n$ | $p$ | ... | $t$|$a$ | $b$ | $c$ |
| Fused Rep |-|-| -| - | $S_2(1, 60) + S_2(61, 63)$ | -|-|- |

Here $S_1(1,3) = S_2(61, 63) = at_c + bt_b+ct_a$, which means that we can reuse the learned partial representations of Seq 1 for Seq 2.
This property is extremely useful, especially when scaling with larger datasets where many users have overlapping sub-sequences.
Note that although architectures such as Transformer can capture sequence-level interactions globally ($v_j = \text{softmax}(Q^\top K_j)V_j$), the attention mechanism within a sequence cannot be disentangled ($v_j \neq S(0, j) + S(j+1, n-1)$) due to the nonlinear nature of attention calculation, thus lacking the generalization for sharing semantic similarities of sub-sequences.

Besides, TedRec can effectively enhance the distinguishability of text representations due to the following two developments:

- **Positional Modulation $W$**. As for text sequence representation $T$, our implemented modulation embedding ($s \in \mathbb{R}^{n \times d}$ in Eq. (4)) in the text adapter and the global learnable embedding $W$ can both be interpreted as the positional modulation for text embeddings ($T + W$). That is to say, we set corresponding positional embedding for each position in the text sequence to better capture the temporal semantic features, while traditional fusion strategies do not separately consider the positional encoding for text sequences.
- **MoE-enhanced text adapter**: has been discussed in part 1.

Overall, our proposed sequence-level semantic fusion method is simple yet effective, which not only provides theoretical analysis but also achieves robust and significant results. We have discovered, proven, and validated our effectiveness, and its extended sequence-level fusion paradigm for different modalities is groundbreaking. Recently, breakthroughs have been made in large language models, thus tapping enormous potential of integrating modal information such as language into recommender systems. TedRec does not rely on the rich information content of the text, can support various model backbones, and has the potential to be applied to multimodal scenes such as images. It has further research significance in the fields of sequential recommendation, side information fusion, language model enhanced recommenders, and multimodal recommendation. 

Thanks again for your valuable comments! If you have further questions about our paper, looking forward to your active discussions! 

Best,

Authors of Submission 707