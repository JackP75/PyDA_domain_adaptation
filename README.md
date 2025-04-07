# PyDA - Domain Adaptation

This repository provides a collection of prominent domain adaptation algorithms implemented in Python. The algorithms are organised into three subpackages: deep (Neural Network-based Method)s, kernel (Kernel-based Methods), and stat (Statistical Alignment Methods). All algorithms can be found in "\TL_models".

## Neural Network-based Methods

- **Domain Adversarial Neural Network (DANN)**
  - [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) (Ganin et al., 2015)

- **Conditional Domain Adversarial Network (CDAN)**
  - [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667) (Long et al., 2017)

- **Domain Adaptation Network (DAN)**
  - [Learning transferable features with deep adaptation networks](https://proceedings.mlr.press/v37/long15.pdf) (Long et al., 2015)

- **Joint Adaptation Network (JAN)**
  - [Deep transfer learning with joint adaptation networks](https://proceedings.mlr.press/v70/long17a/long17a.pdf) (Long et al., 2017)

## Kernel-based Methods

- **Transfer Component Analysis (TCA)**
  - [Domain adaptation via transfer component analysis](https://ieeexplore.ieee.org/document/5640675) (Pan et al., 2010)

- **Joint Distribution Adaptation (JDA)**
  - [Transfer feature learning with joint distribution adaptation](https://openaccess.thecvf.com/content_iccv_2013/papers/Long_Transfer_Feature_Learning_2013_ICCV_paper.pdf) (Long et al., 2013)

- **Balanced Distribution Alignment (BDA)**
  - [Balanced distribution adaptation for transfer learning](https://ieeexplore.ieee.org/document/8215613) (Wang et al., 2017)

- **Geodesic Flow Kernel (GFK)**
  - [Geodesic flow kernel for unsupervised domain adaptation](https://arxiv.org/abs/1301.6708) (Gong et al., 2012)

## Statistical Alignment Methods

- **Correlation Alignment (CORAL)**
  - [Correlation Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/1612.01939) (Sun et al., 2016)

- **Normal Condition Alignment (NCA)**
  - [On Statistic Alignment for Domain Adaptation in Structural Health Monitoring](https://journals.sagepub.com/doi/full/10.1177/14759217221110441) (Poole et al., 2022)

## Installation

To install PyDA, clone the repository and install the required dependencies:

```bash
pip install git+https://github.com/JackP75/PyDA_domain_adaptation.git
```

OR

```bash
git clone https://github.com/JackP75/PyDA_domain_adaptation.git
cd PyDA_domain_adaptation
pip install -r requirements.txt
```
