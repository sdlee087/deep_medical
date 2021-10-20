# Deep_Medical

Original Scripts from Youngwon Choi @Young-won

### Envrionment
- Python 3.6
- Tensorflow 1.15.0
- Keras 2.3.1


“Cheatsheet”
=======================
1. Preprocessing the data (convert raw data to the format readable for python)
1. Add specific reader (easy/no changed needed)
1. Design the architecture and save as '.yaml' file
1. Set configuration file about training hyperparameter and path information
1. Select the number of GPUs and CPU cores from the bash file
1. Run!
1. Launch the tensorboard and track the training situations

Detailed Process
=======================
### 1) Preprocess - Example: __preprocessing_isles.py__, __preprocessing_mri_multicenter.py__
1. If image is small enough, can compile into one file
1. if image is large, then keep separate

### 2) Make lists of identifiers (e.g. patient_id), _x_paths_, _y_paths_ that will be iterated through - Example: x_list.npy

### 3) [Reader] Load image(s) & normalization (if necessary) & processing patches if needed - __readers.py__
1. If 1-1 then load whole file and then load batch-wise
1. If 1-2 then load batch-wise
1. Base custom classes from readers_base.py

### 4) Sampler (weightable)  - samplers.py
- Probability vector/ oversampling (weighting)
- Patch sampling
- Can customize 

---

### 5) Design the architecture (architecture_classification.ipynb)
- if needed, we can add customized keras layer class (__ops.py__)

### 6) Build network
- build network (__build_network.py__)
- model compiler (__build_network.py__)
    - loss function
    - metrics
    - weights
- fit or fit_generator
    - add callback using the configuration files: ()
        - default: BaseLogger (do not need to add)
        - recommendation: 
            - ModelCheckpoint: save the best model for validation data
            - EarlyStopping: stop training when a monitored quantity (based on the validation data) has stopped improving
            - LearningRateScheduler
            - Tensorboard

### 7) loss and metric functions (loss_and_metric.py)

---

### 8) Tensorboard

- provide training logs, network architecture graphic and histograms by default
- If you want to customize the tensorboard output (e.g. embedding space (lower-dimension summary), intermediate output maps), check “set_model” and “on_epoch_end” functions (e.g. TensorBoardSegmentationWrapper in __tensorboard_utils.py__)
    - build image output in set_model of __tensorboard_utils.py__
    - set the execution method in on_epoch_end of __tensorboard_utils.py__

---

### 9) Construct the experiments (results_isles/)
1. Set the configuration details for each experiment (__ipf_seg_*/__)
1. Run the main code (__main_cv.py__)
1. Launch the tensorboard 
    - example with port 6006
    ```tensorboard --logdir=tb_logs --port=6006```
1. Analyze the results
    - Comparison based on notebook (__analysis_CV_ipf_segmentation.ipynb__)
    - Make figures

Useful references
=======================
- Keras Document: https://keras.io/; Github: https://github.com/keras-team/keras
- Keras Examples: https://github.com/keras-team/keras/tree/master/examples
- Kaggle: https://www.kaggle.com/ 
    - recommand to see Notebooks (https://www.kaggle.com/kernels)

