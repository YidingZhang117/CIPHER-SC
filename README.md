# CIPHER-SC

CIPHER-SC is the latest version of CIPHER. It is a complete end-to-end prediction algorithm based on a context-aware network including single-cell data. 
## Prerequisite
Cipher-SC is implemented mainly based on PyTorch and PyTorch Geometric (PyG). So please install these two libraries first.

* [PyTorch Installation](https://pytorch.org/)
* [PyG installation](https://github.com/rusty1s/pytorch_geometric)

## Preprocess
To train and test with CIPHER-SC, we should run preprocess first. The first step is to run `generate_edgelist.py` under `preprocess/edgelist_result`. Then follow the step 1-4 in `process` folder.

After that, context-aware networks and the corresponding training data will be generated under `dataset` folder. Dataset can be downloaded [here](https://drive.google.com/open?id=1995jI2uiWVVvUSYEV41vOIaMbN7hdzcb). You should unzip and place it under the dataset folder, i.e., `dataset/union_SMH_PG`.

## Train
For training Cipher-SC with default setting, you can directly run as follows:

```
python train.py 
```

More parameter choices can be found in `train.py`.

## Test

We provide the checkpoint of Cipher-SC under the `checkpoint` folder, directly run the following code:

```
python test.py
``` 

Then the final result (0.9501 of AUC) in our paper is obtained.