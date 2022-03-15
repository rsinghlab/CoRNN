# CoRNN
Compartment prediction using Recurrent Neural Network


## Setup Comet

pip install comet_ml
pip3 install comet_ml

experiment = Experiment(
    api_key="pTKIV6STi3dM8kJmovb0xt0Ed",
    project_name="cornn-temp",
    workspace="suchzheng2",
)

## link to data

* https://drive.google.com/file/d/1uhM991GNBzJSZoXZhX-uW0FhemFJPhwi/view?usp=sharing

## Sample command

* python cnn/hm2ab.py --data_dir "data/6_cell_input_updated/6_cell_input_updated_100kb/" --task "cla" --model "gru" --split "cross_validation" -Ts --epoch 10 --resolution "100kb" --cross_validation True --add_mean_evec True --num_fold 5 --special_tag "test" --cell "IMR90" --learning_rate 0.01 --hidden 64 --layer 1