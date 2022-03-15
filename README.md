# CoRNN
Compartment prediction using Recurrent Neural Network


## 1. Setup Comet

Note: this program is setup based on comet_ml platform, all training process and statistic can be viewed on comet ml.

1. Install comet in your enviroment
pip install comet_ml
pip3 install comet_ml

2. Sign up on https://www.comet.ml/site/ and create a new project. 
Each project have a specific experiment info. Here is an example of experiment info. 
```python
experiment = Experiment(
    api_key="pTKIV6STi3dM8kJmovb0xt0Ed",
    project_name="cornn-temp",
    workspace="suchzheng2",
)
```

3. Copy the experiment info code from your own project to replace the one in code/hm2ab.py, then you can your experiment stats on your project panel. 

4. (Not recommended) If you run the code directly using the following sample command, your experiment will be upload to this temperaty project: https://www.comet.ml/suchzheng2/cornn-temp/view/new/panels


## 2. link to data

1. Dowload training data from this link: 
* https://drive.google.com/file/d/1uhM991GNBzJSZoXZhX-uW0FhemFJPhwi/view?usp=sharing

2. Unzip it under CoRNN folder, you will have CoRNN/data/

## 3. Sample command

1.  Sample 1 : Train CoRNN on five cells except IMR90 using cross-validation with hidden size 64 and one layer of GRU. Include mean eigenvector in training data.

```
python code/hm2ab.py --data_dir "data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" --model "gru" -Ts --epoch 10 --resolution "100kb" --cross_validation True --add_mean_evec True --num_fold 5 --special_tag "test" --cell "IMR90" --learning_rate 0.01 --hidden 64 --layer 1
```

2. Sample 2 : Grid search for each cell lines using cross-validation, include mean eigenvector in training data
```
* python code/grid_search/grid_search_100kb_GRU_cross_validation_with_mean.py
```

**Command Arguments**
* --data_dir : (str) training data path
* --task:  (str) modeling task type, classification or regression (cla,reg) 
* --model: (str) model type (gru, lstm, cnn1)
* --split: (str) how data split for train valid test
* -T : do train (bool)
* -s: save model (bool)
* --add_mean_evec: (bool) add mean eigenector to training data
* --cell: (str) cell line name that the model want to be tested on (exclude from training)


## dependencies

* python=3.7.1
* comet_ml
* numpy
* tqdm
* pytorch
* sklearn
* matplotlib