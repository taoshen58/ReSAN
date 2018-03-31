# Sentences Involving Compositional Knowledge (SICK)
* Dataset official web site: [http://clic.cimec.unitn.it/composes/sick.html](http://clic.cimec.unitn.it/composes/sick.html)
* The detail of the framework of this project is elaborated at the end of `README.md` in the root of this repo.

## Data Preparation
* Please download the 6B pre-trained model to ./dataset/glove/. The download address is http://nlp.stanford.edu/data/glove.6B.zip
* This repo has contained the dataset. 

**Data files checklist for running with default parameters:**

    ./dataset/glove/glove.6B.300d.txt 
    ./dataset/SICK/SICK.txt


## Training Model

    clone https://github.com/taoshen58/ReSAN
    cd SICK_rl_pub
    python3 sick_rl_main.py --network_type resan --base_name resan --model_dir_suffix training --gpu 0
    

The results will appear at the end of training. We list several frequent use parameters:

* `--num_steps`: training step;
* `--eval_period`: change the evaluation frequency/period.
* `--save_model`: [default false] if True, save model to the model ckpt dir;
* `--train_batch_size` and `--test_batch_size`: set to smaller value if GPU memory is limited.
* `--dropout` and `--wd`: dropout keep prob and L2 regularization weight decay.
* `--word_embedding_length` and `--glove_corpus`: word embedding Length and glove model.
* `--load_model`: if a pre-trained model is needed for further fine-tuning, set this parameter to True
* `--base_name`: mode base name
* `--pretrained_path`: if `--load_model` is set to True, the pre-trained model path. If this value is None, the path will be set to *./pretrained_model/${base_name}.ckpt*;
* `--start_only_rl`: the step to start only reinforcement learning with stable env provided;
* `--end_only_rl`: the step to end only reinforcement learning and start training RL and SL simultaneously;
* `--rl_sparsity`: reinforcement sparsity parameter and we found 0.005~0.0075 is good for SICK dataset.

    


