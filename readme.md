# CClinguist

An Expert-Free Framework for Future-Compatible Congestion Control Algorithm Identification

## Dependencies

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

4. **Python Packages**
   - Use pip to install the following packages: `torch`, `sklearn`, `numpy`, `pandas`, `dtw-python`, `math`.

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
2. Modify the `data_path` in `classifier/classifier.py` to point to your data.
3. Train the classifier by running:
    ```sh
    python Classifier/classifier.py
    ```
4. With the trained classifier, you can train the profile generator. Make sure to set the options in `profile_generator/option.py`:
    ```sh
    python profile_generator/run.py
    ```

### 2. Testing

Given a trained Classifier and Generator, use the following script to test the example data:
```sh
python profile_generator/run.py
```

Analyze the results with:
```sh
python profile_generator/results_analyze.py
```


## Other SOTA Work

Implementations of other state-of-the-art (SOTA) work:

- [CCAanalyzer](https://dl.acm.org/doi/pdf/10.1145/3651890.3672255)
- [Nebby](https://dl.acm.org/doi/pdf/10.1145/3651890.3672223)
