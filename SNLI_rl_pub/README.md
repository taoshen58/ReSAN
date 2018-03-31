# Stanford Natural Language Inference (SNLI)
* Dataset official web site: [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
* The detail of the framework of this project is elaborated at the end of `README.md` in the root of this repo.

## Data Preparation
* Please download the 6B pre-trained model to ./dataset/glove/. The download address is http://nlp.stanford.edu/data/glove.6B.zip
* Please download the SNLI dataset files from [https://nlp.stanford.edu/projects/snli/snli_1.0.zip](https://nlp.stanford.edu/projects/snli/snli_1.0.zip).

**Data files checklist for running with default parameters:**

    ./dataset/glove/glove.6B.300d.txt
    ./dataset/snli_1.0/snli_1.0_train.jsonl
    ./dataset/snli_1.0/snli_1.0_dev.jsonl
    ./dataset/snli_1.0/snli_1.0_test.jsonl
    
    
## Test with Pre-trained Model

First, clone the code to local by running:

    clone https://github.com/taoshen58/ReSAN
    cd SNLI_rl_pub

Then download the data pre-processing file and pre-trained model at [Google Drive](https://drive.google.com/open?id=10wP4Vr_4QizCXj3pJQcvkFE5sRqnNwfY)

Put the files to corresponding folders. **Files Checklist:** 
    
    ./pretrained_model/snli_pretrained_model.ckpt.data-00000-of-00001
    ./pretrained_model/snli_pretrained_model.ckpt.index
    ./result/processed_data/processed_lw_True_ugut_True_gc_6B_wel_300_slr_0.97_dcm_no_tree.pickle

And simply Run:


    python3 snli_main.py --mode test --network_type hw_resan --base_name hw_resan --load_model True --load_path ./pretrained_model/snli_pretrained_model.ckpt --test_batch_size 100 --model_dir_suffix test_mode --gpu 0

Note if the running performance does not match the result we report in the paper, please re-run the command above due to the randomness in sampling.

## Model Training
    
    clone https://github.com/taoshen58/ReSAN
    cd SNLI_rl_pub
    python3 snli_main.py --mode train --network_type hw_resan --rl_sparsity 0.02 --model_dir_suffix training --gpu 0


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
* `--rl_sparsity`: reinforcement sparsity parameter and we found 0.02~0.04 is good for SNLI dataset.

