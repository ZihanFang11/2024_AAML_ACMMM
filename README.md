# Beyond the Known: Ambiguity-Aware Multi-view Learning

<h2 align="center"><a href="https://dl.acm.org/doi/10.1145/3664647.3681362">Beyond the Known: Ambiguity-Aware Multi-view Learning</a></h2>

<div align="center">

**Zihan Fang<sup>1</sup>, Shide Du<sup>1</sup>, Yuhong Chen<sup>1</sup>, Shiping Wang<sup>1</sup>**

<sup>1</sup>Fuzhou University, Fuzhou, China<br>
</div>

## Abstract
The inherent variability and unpredictability in open multi-view learning scenarios infuse considerable ambiguity into the learning and decision-making processes of predictors. This demands that predictors not only recognize familiar patterns but also adaptively interpret unknown ones out of training scope. To address this challenge, we propose an Ambiguity-Aware Multi-view Learning Framework, which integrates four synergistic modules into an end-to-end framework to achieve generalizability and reliability beyond the known. By introducing the mixed samples to broaden the learning sample space, accompanied by corresponding soft labels to encapsulate their inherent uncertainty, the proposed method adapts to the distribution of potentially unknown samples in advance. Furthermore, an instance-level sparse inference is implemented to learn sparse approximated points in the multiple view embedding space, and individual view representations are gated by view-level confidence mappings. Finally, a multi-view consistent representation is obtained by dynamically assigning weights based on the degree of cluster-level dispersion. Extensive experiments demonstrate that our approach is effective and stable compared with other state-of-the-art methods in open-world recognition situations.

## Model Architecture

<div align="center">
  <img src="framework.png" alt="AAML framework" />
</div>

### Datasets Preparation

The experiments use six multi-view benchmarks:

| Dataset | # Samples | Feature dimensions by view | # Classes | Type |
|---|---:|---|---:|---|
| ESP-Game | 11,032 | 100 / 100 | 7 | Multi-view image/game data |
| Flower17 | 1,360 | 1,360 / 1,360 / 1,360 / 1,360 / 1,360 / 1,360 / 1,360 | 17 | Image data |
| MNIST | 2,000 | 1,930 / 9 / 30 | 10 | Digit images |
| Reuters | 1,500 | 21,531 / 24,892 / 34,251 / 15,506 / 11,547 | 6 | Text documents |
| ORL | 400 | 512 / 59 / 864 / 254 | 40 | Face images |
| Youtube | 2,000 | 2,000 / 1,024 / 64 / 512 / 64 / 647 | 10 | Video features |

Place the corresponding `.mat` files in the data directory expected by the training script:

```text
data/
├── esp_game.mat
├── flower17.mat
├── MNIST.mat
├── ORL.mat
├── smallReuters.mat
└── Youtube.mat
```

### Usage

```bash
python test.py
```

### Evaluation

Open-set performance is reported with the Open-Set Classification Rate (OSCR), using CCR at FPR values of `0.5%`, `1.0%`, `5.0%`, and `10.0%`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fang2024beyond,
  title={Beyond the Known: Ambiguity-Aware Multi-view Learning},
  author={Fang, Zihan and Du, Shide and Chen, Yuhong and Wang, Shiping},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={8518--8526},
  year={2024},
  doi={10.1145/3664647.3681362}
}
```
