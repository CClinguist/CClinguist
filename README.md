# CClinguist: An Expert-Free Framework for Future-Compatible Congestion Control Algorithm Identification
---
## 👥 Authors
- Jiahui Li(Fudan University) <22110240094@m.fudan.edu.cn>  
- Han Qi (Fudan University) <hqi23@m.fudan.edu.cn>  
- Ruyi Yao (Fudan University) <ryyao20@fudan.edu.cn> 
- Jialin Wei (Fudan University) <24210240332@m.fudan.edu.cn>  
- Ruoshi Sun (Fudan University) <rssun23@m.fudan.edu.cn> 
- Zixuan Chen (Fudan University) <zxchen20@fudan.edu.cn>
- Sen Liu (Fudan University) <senliu@fudan.edu.cn>
- [Yang Xu](https://yangxu.info/) (Fudan University) <xuy@fudan.edu.cn>  

## 📜 Publication
This work is published in ACM SIGCOMM 2025:  
**CClinguist: An Expert-Free Framework for Future-Compatible Congestion Control Algorithm Identification**  

## Overview

**CClinguist** consists of three main components:

1. **Data Collection**
2. **Classifier**
3. **Profile Generator**

The framework collects data, trains a classifier, and trains a profile generator to identify Congestion Control Algorithms (CCAs).  
We provide pre-trained models for quick testing, you can also collect your own data and retrain the models.

---

## A. Docker Setup

For detailed instructions, see [README_docker.md](README_docker_en.md).

---

## B. Testing

### 1. Identification of 12 Linux CCAs

1. **Download** `data_12ccs` from `https://drive.google.com/drive/folders/1ITS85xJCkCdJ9o5Pi9c6OdUuwER1MLTx?usp=sharing` to:

    ```
    profile_generator_12ccs/data_12ccs
    ```

2. **Run**:

    ```bash
    cd profile_generator_12ccs
    bash test_from_params.sh
    ```

3. **View Results**:  
   - Analysis: `results_12CCs_coef/analyze_final`  
   - Decision tree: `tree_visualization`

---

### 2. Identification of 12 Linux CCAs + 3 New CCAs

1. **Download** `data_15ccs` from `https://drive.google.com/drive/folders/1ITS85xJCkCdJ9o5Pi9c6OdUuwER1MLTx?usp=sharing` to:

    ```
    profile_generator_15ccs/data_15ccs
    ```

2. **Run**:

    ```bash
    cd profile_generator_15ccs
    bash test_from_params.sh
    ```

3. **View Results**:  
   - Analysis: `results_15CCs_coef/analyze_final`  
   - Decision tree: `tree_visualization`

---

### 3. Self-Upgrading for New CCAs

1. **Download Dataset** `data_upgrading` from `https://drive.google.com/drive/folders/1ITS85xJCkCdJ9o5Pi9c6OdUuwER1MLTx?usp=sharing` to:

    ```
    self_upgrading_unknown_similar/data_upgrading
    ```

2. **Run**:

    ```bash
    cd self_upgrading_unknown_similar
    python main_unknown.py
    ```

3. **View Results**:  
   - Output in `dis_log`
   - CCA library sizes: 12, 13, 14, and 15 CCAs
   - Servers deployed with Astraea, PCCLoss, and PCCLatency are identified as specific known CCAs, along with distance metrics.

---

### 4. Identification of Similar CCAs

1. **Run**:

    ```bash
    cd self_upgrading_unknown_similar
    python main_simccas.py
    ```

2. **View Results**:  
   - Output in `dis_log_simccas`
   - CCA library sizes:  
     - 13 CCAs (`12 + PCC`)  
     - 15 CCAs (`12 + PCCLoss + PCCLatency + Astraea`)
   - Servers deployed with `pcc_variant1` and `pcc_variant2` are identified as specific known CCAs, along with distance metrics.

---

## C. Data Collection & Training

### Dependencies

1. **Mahimahi**
   - Installation instructions: [Mahimahi](http://mahimahi.mit.edu/)

2. **XDP**
    - sudo apt install -y bison build-essential cmake flex git libedit-dev pkg-config libmnl-dev \
   python zlib1g-dev libssl-dev libelf-dev libcap-dev libfl-dev llvm clang pkg-config \
   gcc-multilib luajit libluajit-5.1-dev libncurses5-dev libclang-dev clang-tools

    
3. **Wandb**
   - Log into [wandb.ai](http://wandb.ai/) (Learn how to deploy a W&B server locally: [wandb-server](https://wandb.me/wandb-server)).
   - You can find your API key in your browser here: [Wandb Authorization](https://wandb.ai/authorize).
   - Paste your API key from your profile.

## Running

### 1. Data Collection

- For data collection, run:
    ```sh
    python data_collection/run.py
    python data_collection/ecn_enable/run.py
    ```

### 2. Training with Data

You can train the classifier and the profile generator with the data you collected. Follow these steps:

1. Log in to Wandb with your API key.
2. Modify the `data_path` in `classifier/train.py` to point to your data.
3. Train the classifier by running:
    ```sh
    python classifier/train.py
    ```
4. With the trained classifier, you can train the profile generator. Make sure to set the options in `profile_generator/option.py`.

## D. Visualization Scripts

The `figs/` folder contains visualization scripts for generating figures used in the paper:

**Usage**:
```bash
cd figs/dis_scatter
python plot_dis_scatter.py
python plot_simccas_scatter.py


cd figs/trace
python plot.py
python plot_grid.py
```

**Dependencies**:
- `matplotlib`
- `pandas`
- `numpy`

---

## Other SOTA Work

Implementations of other state-of-the-art (SOTA) work:

- [CCAanalyzer](https://dl.acm.org/doi/pdf/10.1145/3651890.3672255)
- For [Nebby](https://dl.acm.org/doi/pdf/10.1145/3651890.3672223), refer to the [official implementation](https://github.com/NUS-SNL/Nebby) provided by the authors.

