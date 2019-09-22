# Model Stability

This is the code for the paper: `On Model Stability as a Function of Random Seed` (adding link to the paper soon!) by Rishabh Jain and Dr. Pranava Madhyastha.
It is forked from the [AttentionExplanation](https://github.com/successar/AttentionExplanation) repository.

To run the experiments follow the steps mentioned in the [Prerequisites](#prerequisites), [Update](#update), [Binary Classification Tasks](Binary-Classification-Tasks) and other sections.

To run the stability experiments with `NASWA`, you can run the following:
`python3 train_and_run_stability_test_bc.py --dataset sst --output_dir outputs/ --attention dot --encoder lstm --swa [1,1,1,1];`

To run the stability experiments with `ASWA`, you can run the following:
`python3 train_and_run_stability_test_bc.py --dataset sst --output_dir outputs/ --attention dot --encoder lstm --swa [1,1,1,0];`

To run the stability experiments without `NASWA/ASWA`, you can run the following:
`python3 train_and_run_stability_test_bc.py --dataset sst --output_dir outputs/ --attention dot --encoder lstm --swa [0,0,0,0];`
 
Running these scripts will train the model with 10 seeds and save the attention based explanations in a pickled file.
You can use the `train_and_run_lime.py` to generate the lime based explanations as well.


## Prerequisties

This project requires compiling `pytorch` from source master branch or use `pytorch-nightly`. We use features that are not in stable release. It also requires installation of torchtext version 0.4.0 from source.

After installation of above, please use `pip install -r requirements.txt`.
Also, `python -m spacy download en` to include the english language pack for spacy if not already present.

## Update

We are providing code to run experiments on all datasets . For obtaining ADR tweets data, please contact us directly (a large portion of tweets we have used in this experiments have been removed from twitter website).

1. Clone the repository as `git clone https://github.com/successar/AttentionExplanation.git Transparency` (Note this is important.)

2. Set your PYTHONPATH to include the directory path which contains this repository (All imports in the code are of form Transparency.* -- If you see error `ModuleNotFoundError: No module named 'Transparency'`, most probably your PYTHONPATH is not set.). 

For example if your cloned repository reside in `/home/username/Transparency`, then one way to do this is `export PYTHONPATH="/home/username"` from command line or add it to your `~/.bashrc` .

3. Go to the `Transparency/preprocess` folder and follow the instructions to process datasets.

## Binary Classification Tasks

1. From the main folder, run `python train_and_run_experiments_bc.py --dataset {dataset_name} --data_dir . --output_dir outputs/ --attention {attention_type} --encoder {encoder_type}`

Valid values for `dataset_name` are  `[sst, imdb, 20News_sports, tweet, Anemia, Diabetes, AgNews]`.

Valid values for `encoder_type` is `[cnn, lstm, average]`.
Valid values for `attention_type` is `[tanh, dot]`.

For example, if you want to run experiments for IMDB dataset with CNN encoder and Tanh attention, please use `python train_and_run_experiments_bc.py --dataset imdb --data_dir . --output_dir outputs/ --attention tanh --encoder cnn`

To run QA or SNLI tasks,
------------------------

1. From the main folder, run `python train_and_run_experiments_qa.py --dataset {dataset_name} --data_dir . --output_dir outputs/ --attention {attention_type} --encoder {encoder_type}`

Valid values for `dataset_name` are  `[snli, cnn, babi_1, babi_2, babi_3]`.

Valid values for `encoder_type` is `[cnn, lstm, average]`.
Valid values for `attention_type` is `[tanh, dot]`.

For example, if you want to run experiments for snli dataset with LSTM encoder and Tanh attention, please use `python train_and_run_experiments_bc.py --dataset snli --data_dir . --output_dir outputs/ --attention tanh --encoder lstm`
