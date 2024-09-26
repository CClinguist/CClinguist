# CClinguist

An Expert-Free Framework for Future-Compatible Congestion Control Algorithm Identification

## Dependencies

1. **Mahimahi**
   - Installation instructions: [Mahimahi](http://mahimahi.mit.edu/)
    
2. **Wandb**
   - Log into [wandb.ai](http://wandb.ai/) (Learn how to deploy a W&B server locally: [wandb-server](https://wandb.me/wandb-server)).
   - You can find your API key in your browser here: [Wandb Authorization](https://wandb.ai/authorize).
   - Paste your API key from your profile.

3. **Python Packages**
   - Use pip to install the following packages: `torch`, `sklearn`, `numpy`, `pandas`, `dtw-python`, `math`.

## Running

### 1. Data Collection

- For testbed data collection, run:
    ```sh
    CClinguist/DataCollection/testbed/run_with_capture_simulation.sh
    ```
    
- For online data collection, run:
    ```sh
    CClinguist/DataCollection/Online/run_with_capture_realUrls.sh
    ```

### 2. Testing

Given a trained Classifier and Generator, use the following script to test the example data:
```sh
python CClinguist_source/CClinguist/Profile_generator/run.py
```

Analyze the results with:
```sh
python CClinguist_source/CClinguist/Profile_generator/results_analyze.py
```

### 3. Training with Your Data

You can train the classifier and the profile generator with the data you collected. Follow these steps:

1. Log in to Wandb with your API key.
2. Modify the `data_path` in `CClinguist_source/CClinguist/Classifier/classifier.py` to point to your data.
3. Train the classifier by running:
    ```sh
    python CClinguist_source/CClinguist/Classifier/classifier.py
    ```
4. With the trained classifier, you can train the profile generator. Make sure to set the options in `CClinguist/Profile_generator/option.py`:
    ```sh
    python CClinguist_source/CClinguist/Profile_generator/run.py
    ```

## Other SOTA Work

Implementations of other state-of-the-art (SOTA) work:

- [CCAanalyzer](https://dl.acm.org/doi/pdf/10.1145/3651890.3672255)
- [Nebby](https://dl.acm.org/doi/pdf/10.1145/3651890.3672223)