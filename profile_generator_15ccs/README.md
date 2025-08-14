# READMD

# 1.Accuracy Validation Experiment

To reproduce accuracy validation results:

0. Download data_15ccs from https://drive.google.com/drive/folders/1ITS85xJCkCdJ9o5Pi9c6OdUuwER1MLTx?usp=sharing to  profile_generator_15ccs/data_15ccs
1. Open `test_from_params.sh`
2. Set:
    
    ```bash
    CONFIG_FILE="params_test_base.json"
    ```
    
3. Run:
    
    ```bash
    cd  profile_generator_15ccs/data_15ccs
    bash test_from_params.sh
    ```
    
4. See the results in results_15CCs_coef/analyze_final and the tree in tree_visualization

# 2.Parameter-Independent Experiment

If you only want to verify **accuracy-independent** results (pre-generated parameters):

1. Open `test_from_params.sh`
2. Set:
    
    ```bash
    CONFIG_FILE="params_test.json
    ```
    
3. Run:
    
    ```bash
    bash test_from_params.sh
    ```
    

# 3.Parameter-Independent Experiment (Re-validation with Random Parameters)

If you want to **re-generate random parameters** and re-run experiments:

1. Generate random parameters:
    
    ```bash
    python param_generator.py
    ```
    
    This will produce a `params_train.json` file.
    
2. Switch to **train mode** in `option_newcca.py`:
    
    modify flag to 1
    ```python
    flag = 1  # train mod
    ```
    
    (By default, `flag = 0` means test mode.)
    
3. Run training:
    
    ```bash
    bash train_from_params.sh
    
    ```
    

## Notes

- **flag in `option_newcca.py`**:
    - `flag = 0` → Test mode (default)
    - `flag = 1` → Train mode (used for re-generating parameters)
- Please ensure all `.json` configuration files are placed in the root directory before execution.

Note: Modify the path/parameter settings in both scripts before running them.