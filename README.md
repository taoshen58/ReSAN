# Reinforced Self-Attention Network (ReSAN)

* The repository is the experimental codes for [Title]()
* All codes is implemented with Python under Deep Learning framework Tensorflow.
* The leaderboard of Stanford Natural Language Inference is available [here](https://nlp.stanford.edu/projects/snli/).
* Please contact author or open an issue for questions and suggestions.

## Requirements
### Foundation
* Python3 (verified on 3.5.2, Anaconda3 4.2.0)
* tensorflow>=1.2

### Other Python Packages
* tqdm
* nltk (please download Models/punkt)


------
## Contents of this Repository
1. Standalone codes for Reinforced Sequence Sampling (RSS), Reinforced Self-Attention (ReSA) and Reinforced Self-Attention Network (ReSAN), which are detailed in the paper. --- dir: **resan**
2. Projects codes for the experiments on the Stanford Natural Language Inference (SNLI) Dataset. --- dir: **SNLI_rl_pub**
3. Projects codes for the experiments on the Sentences Involving Compositional Knowledge (SICK) Dataset. --- dir: **SICK_rl_pub**

**For the first one, we will give APIs to introduce how to use it at the rest of this file. For others, please enter corresponding directory for further instructions.**

----------

### APIs

#### Reinforced Sequence Sampling (RSS)

    from resan.rl_nn import generate_mask_with_rl
    

##### Parameters:

* param **rep_tensor**: [tf Float 3D Tensor] The rank must be 3, i.e., [batch_size, seq_len, feature_dim];
* param **rep_mask**: [tf Bool 2D Tensor] The mask for input with rank 2: [batch_size, seq_len];
* param **is_mat**: [Python Bool]sampling a sequence(False) or a matrix(True), Keep *False*
* param **scope**: [str] The name of Tensorflow variable scope;
* param **keep_prob**: [float] dropout keep probability;
* param **is_train**: [tf Bool Scalar] the indicator of training or test;
* param **wd**: [float >= 0] if wd>0, add related tensor to tf collectoion "weight_decay" for further l2 decay;
* param **activation**: [str] in (relu|elu|slu);
* param **disable_rl**: [Python Bool] disable the rl by forcing all probabilities to 1 if *True*;
* param **global_step**: [Python int] global training step, if it is less than **start_only_rl**, all probabilities is forced to set to 1;
* param **mode**: [str] indicating current status of projects in training, test, etc. phrase. If the **mode** is not 'train', the reinforcement learning will be forced to open regardless of other parameters;
* param **start_only_rl**: [Python int] the global step to start training hard network;
* param **hn**: Hidden units number

##### Returns:
* **logpa** [TF float matrix] with shape (batch_size, seq_len) to indicate logarithmic selection probabilities;
* **actions** [TF bool matrix] with shape (batch_size, seq_len) to indicate the selection or not by sampling
* **percentage**: [TF float vector] with shape (batch_size) to indicate the selection percentage for sampling sparsity.


#### Reinforced Self-Attention (ReSA)

    from resan.resa import reinforced_self_attention
    

##### Parameters:
* param **rep_tensor**: [tf Float 3D Tensor] The rank must be 3, i.e., [batch_size, seq_len, feature_dim];
* param **rep_mask**: [tf Bool 2D Tensor] The mask for input with rank 2: [batch_size, seq_len];
* param **dep_selection**: [tf Bool 2D Tensor] with shape (batch_size, seq_len), dependent selections from `generate_mask_with_rl`, i.e., RSS for dependent;
* param **head_selection**:  [tf Bool 2D Tensor] with shape (batch_size, seq_len), head selections from `generate_mask_with_rl`, i.e., RSS for head;
* param **hn**: hidden units number
* param **keep_unselected**: [Python bool] whether keep the unselected head, which introduced in the paper as the variant;
* param **scope**: [str] The name of Tensorflow variable scope;
* param **keep_prob**: [float] dropout keep probability;
* param **is_train**: [tf Bool Scalar] the indicator of training or test;
* param **wd**: [float >= 0] if wd>0, add related tensor to tf collectoion "weight_decay" for further l2 decay;
* param ***activation**: [str] in (relu|elu|slu);

##### Returns:
a tensorflow tensor with shape (batch_size, seq_len, 2*hn) to denotes the context-aware representations for all tokens.

#### Reinforced Self-Attention Network (ReSAN)

    from resan.resan import reinforced_self_attention_network
    

##### Parameters:
same as `reinforced_self_attention` introduced above
##### Returns:
a tensorflow matrix with shape (batch_size, 2*hn) to denotes the all sequence/sentence encodings



--------
-------
## Programming Framework for SNLI or SICK

We first demonstrate the file directory tree of all these projects:

```
ROOT
--dataset[d]
----glove[d]
----$task_dataset_name$[d]
--pretrained_model [d]
--src[d]
----model[d]
------template.py[f]
------$model_name$.py[f]
----nn_utils[d]
----utils[d]
------file.py[f]
------nlp.py[f]
------record_log.py[f]
------time_counter.py[f]
----dataset.py[f]
----evaluator.py[f]
----graph_handler.py[f]
----perform_recorder.py[f]
--result[d]
----processed_data[d]
----model[d]
------$model_specific_dir$[d]
--------ckpt[d]
--------log_files[d]
--------summary[d]
--------answer[d]
--configs.py[f]
--$task$_main.py[f]
--$task$_log_analysis.py[f]
```

Note: The result dir will appear after the first running.

We elaborate on the every files[f] and directory[d] as follows:

`./configs.py`: perform the parameters parsing and definitions and declarations of global variables, e.g., parameter definition/default value, name(of train/dev/test_data, model, processed_data, ckpt etc.) definitions, directories(of data, result, `$model_specific_dir$` etc.) definitions and corresponding paths generation. 

`./$task$_main.py`: this is the main entry python script to run the project;

`./$task$_log_analysis.py`: this provides a function to analyze the log file of training process. 

`./dataset/`: this is the directory including datasets for current project.

* `./dataset/glove`: including pre-trained glove file
* `./dataset/$task_dataset_name$/`: This is the dataset dir for current task, we will concretely introduce this in each project dir.

`./pretrained_model/`: this is the directory including the default pretrained supervised learning network checkpoints for corresponding base name.

`./src`: dir including python scripts

* `./src/dataset.py`: a class to process raw data from dataset, including data tokenization, token dictionary generation, data digitization, neural network data generation. In addition, there are also some method: `generate_batch_sample_iter` for random mini-batch iteration, `get_statistic` for sentence length statistics and a interface for deleting samples with long sentence in training data.
* `./src/evaluator.py`: a class for model evaluation.
* `./src/graph_handler.py`: a class for handling graph: session initialization, summary saving, model restore etc.
* `./src/perform_recoder.py`: a class to save top-n dev accuracy model checkpoint for future loading.
* `./src/model/`: the dir including tensorflow model file
* `./src/model/template.py`: a abstract python class, including network placeholders, global tensorflow tensor variables, TF loss function, TF accuracy function, EMA for learnable variables and summary, training operation, feed dict generation and training step function. 
* `./src/model/$model_name$.py`: Main TF neural network model, implement the abstract interface `build_network`, extending from `template.py`.
* `./src/nn_utils/`: a package include various tensorflow layers implemented by this repo author.
* `./src/utils/file.py`: file I/O functions.
* `./src/utils/nlp.py`: natural language processing functions.
* `./src/utils/record_log.py`: a log recorder class, and a corresponding instance for all use in current project.
* `./src/utils/time_counter.py`: time counter class to collect the training time, note this time exclude the process to prepare data but only the time spending in training step.

`./result/`: a dir to place the results.

* `./result/processed_data/`: a dir to place Dataset instance with pickle format. The file name is generated by `get_params_str` in `./config.py` according to the related parameters.
* `./result/model/$model_specific_dir$/`: the name of this dir is generated by `get_params_str` in `./config.py` according to the related parameters, to save the result for a combination of the parameters. In other words, we associate a dir for a combination of the parameters.
* `./result/model/$model_specific_dir$/ckpt/`: a dir to save top-n model checkpoints.
* `./result/model/$model_specific_dir$/log_files/`: a dir to save log file.
* `./result/model/$model_specific_dir$/summary/`: a dir to save tensorboard summary and tensorflow graph meta files.
* `./result/model/$model_specific_dir$/answer/`: a dir to save extra prediction result for a part of these projects.

## Parameters in config.py

* `--network_type`: [str] use "exp_context_fusion" for reproduce our experiments
* `--log_period`: [int] step period to save the summary
* `--eval_period`: [int] step period to evaluate the model on dev dataset;
* `--gpu`: [int] GPU index to run the codes;
* `--gpu_mem`: [None or float] if None, it allow soft memory placement, or fixed proportion of the total memory;
* `--save_model`: [Bool] whether save the top-3 eval chechpoints;
* `--mode`: [str] all projects have "train" model to train model, only some projects have "test" model for test, please refer the main script of every project;
* `--load_model`: [Bool] if has "test" mode, set this to True to load checkpoint for test by specify the path-to-ckpt in `--load_path`;
* `--model_dir_suffix`: [str] a name for the model dir (the model dir is a part of programming framework which will be introduced latter);
* `--num_steps`: [int] the training step for mini-batch SGD;
* `--train_batch_size`: [int] training batch size
* `--test_batch_size`:  [int] dev and test batch size. Note that in squad, use `--test_batch_size_gain` instead of it: test_batch_size = test_batch_size_gain * train_batch_size;
* `--word_embedding_length`: [int] word embedding length to load glove pretrained models, which could be 100|200|300 depending on the data provided
* `--glove_corpus`: [str] GloVe corpus name, which could be 6B|42B|840B etc.;
* `--dropout`: [float] dropout keep probability;
* `--wd`: L2 regularization decay factor;
* `--hidden_units_num`: hidden units num. Note that we fix the hidden units num for all context_fusion models on all projects except squad.
* `--optimizer`: [str] mini-batch SGD optimizer, could be adam|adadelta|rmsprop;
* `--learning_rate`: [float] initial learning rate for optimizer;
* `--base_name`: mode base name
* `--pretrained_path`: if `--load_model` is set to True, this indicates the pre-trained model path. If this value is None, the path will be set to *./pretrained_model/$base_name$.ckpt*;
* `--start_only_rl`: the step to start only reinforcement learning with stable env provided;
* `--end_only_rl`: the step to end only reinforcement learning and start training RL and SL simultaneously;
* `--rl_sparsity`: reinforcement sparsity parameter and we found 0.02~0.04 is good for SNLI dataset.
* `--rl_strategy`: [sep|sim] *sep* means that the hard network and other network are trained separately, and *sim* means the two networks are trained simultaneously.
* `--step_for_rl`: when `--rl_strategy` is set to *sep*, the training step for hard network every time.
* `--step_for_sl`: when `--rl_strategy` is set to *sep*, the training step for other network every time.



------
------

**Note** that due to too many codes this repo includes, it is inevitable that there are some wrong when I organize the projects into this repo. If you confront some bugs or errors when running the codes, please feel free to report them by opening a issues. I will reply it ASAP.

## Acknowledge

