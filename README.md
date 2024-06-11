# IRM

## Introduction

The Injectable Realignment Model (IRM) is a trainable feed-forward neural network that modifies a language model's forward pass as it runs, in order to realign the language model's output behavior. This codebase features the initial implementation used to produce the data found in our paper, The Mysterious Case of Neuron 1512, where we introduced the IRM architecture. 

This code performs various preparatory tasks as well as replicates our paper's experiements. To prepare an untrained IRM for learning, the weights of a pretrained Llama-2-7b-chat-hf model are loaded into a LlamaForCausal object that has been modified to also contain a default IRM model; the modified object is an instance of the InjectedLlamaForCausal class. The combined pretrained Llama weights and IRM weights are saved as a checkpoint file. 

To train an IRM with a given alignment, this checkpoint is loaded into the InjectedLlamaForCausal object and trained on data that features the desired alignment. On each forward pass, the IRM takes as input the activations of the zeroth attention layer of the Llama transformer, and outputs a tensor which is summed into the Llama activations at each of the layers specified in the configuration file. Because the pretrained model weights are locked and will remain unaltered by the training process, the IRM weights receive the entirety of the training updates, calculated according to the loss function.

When training is finished, the IRM will now be aligned to the text it was trained on, and will reflect that alignment in the language model's output. Our code generates injected realignment outputs from trained IRMs and produces heatmaps that visualize the output tensors of the IRM at each forward pass, which correspond to the amount of alteration given to the activations of the Llama model when processing each token.

### Workflow

0. Set up your Environment
1. Prepare a Base Model
2. Get Training Data
3. Create a Config File
4. Tokenize the Training Data
5. Create an Injected Checkpoiint
6. Run the training Script
7. Run Inference
8. Analysis

## Setup

### Environment Setup

Create a Mamba environment with python=3.9, preferably named rocket: mamba create -n rocket python=3.9

If it is named differently, the environment activation commands in the Slurm scripts must be changed.

Run pip install -r requirements.txt.

### Getting a Base model

The code in this repo was written to use Llama 2 weights that are saved locally.  The corresponding code that defines the model architecture is also used to specify where in the model the IRM should be injected.

Other methods (such as loading a model from Hugging Face every time and making dynamic changes to its members, or "hijacking") may preclude the following steps needed to save model weights locally and modify the model's code.

#### Model Weights and Corresponding Model Code

Here are some different ways to save weights locally:

*Note: We found that the chat versions of the models worked more effectively than the non-chat ones*

* Get the weights directly from the source:
    * For Llama 2, you can follow the instructions on [Meta's website for Llama 2](https://llama.meta.com/llama2/) to download the Llama 2 weights, which will include requesting access and following their instructions to download the weights.
    * The code that defines Llama 2 architecture will be downloaded by the script

* Instantiate a Hugging Face model and save it:
    * The hf_to_compute() function in [convert_checkpoint.py](./src/convert_checkpoint.py) will save the weights locally.
        1. Change `pretrained_name` to the Hugging Face identifier string.
        2. Change `new_original_checkpoint_path` to the filepath at which the checkpoint will be stored
        3. Call the function from a python3 interpreter. From the /src/ directory:
            1. `python3`
            2. `from convert_checkpoint import *`
            3. `hf_to_compute()`

            - You may need to write a short python script and run it in a slurm job if you run out of memory in the terminal

    * The code that defines the pretrained LlamaForCausal models is found in the [llama_models](./src/llama_models) directory.
    
        If you would like to use a different pretrained Hugging Face model, you'll have to track down the corresponding code in [their repo](https://github.com/huggingface/transformers).  It's huge, so using the search bar is very helpful.  That may also require modifying the files to directly import modules (`from transformers import #####`).

### Setting up a Config file

Configuration YAML files are used to define all paths, settings, and hyperparameters for training tokenizers, tokenizing data, training models, injecting an IRM, and running inference. You can create a new config by copying default_config.yaml (preferebly into the [configs](./configs/) folder) or by using the config generation script as described below.

#### Config Creation Script
There are a lot of details that need to be aligned in the config files, and the [config creation script](./src/utils/new_config_create.py) makes it easier to get all the details right when creating a large number of scripts.

This script is useful if you need to make many different scripts that are slightly different (for example, training several IRMs, each injected in different places, each on several datasets), but if you are only running one model, it might be easier to just change one config file.

To use this script:

* Change the file as specified below.
* Run `python3 ./src/utils/new_config_create` from `.`
* The config file(s) should appear in the [configs/](./configs/) directory.

In the `main` function:

1. Specify `model_name` as the model you'd like to inject into, which should correspond to an entry in the `model_size_directories` dictionary (located in the `create_config_dict` function).

2. Specify `file_name_prefix` as the first part of the name of the config file, checkpoint, and logger folder.

*Note: The ith index in each of the following lists corresponds to the ith config file that will be generated*

3. Specify the injection layers in `injection_locations`.

4. Specify your train, test, and val dataset files in their `####_dataset_file_names` lists.  These should end in .pkl, even if they are not tokenized yet (the script will write .csv and .pkl in their corresponding places in preparation for tokenization).

5. Specify the number of epochs for each corresponding config file in `dataset_file_epochs`

In the `create_config_dict` function:

1. Ensure the `model_size_directories` dictionary maps each model name to the directory where its model weights are stored

2. Specify `tokenizer_type` and `tokenizer_path`:
    * To use locally saved tokenizer weights, change `tokenizer_path` to the filepath of your tokenizer weights and  `tokenizer_type` to `"sp"` (for SentencePiece)
    
        OR
    * To use a Hugging Face tokenizer, change `tokenizer_path` to the Hugging Face tokenizer identifier and  `tokenizer_type` to `"hf"`

3. Ensure that each of the dataset paths leads to the correct directory.

4. Update the settings under #GPU to match the settings in your slurm scripts

5. Update any other settings, including batch size, inference path, or validation settings.

If you are using a model other than the Llama models for which HFConfigs are included in the `get_HF_config` function, you may need to add the config.  This can be done by printing out the model.config member of an instantiated model.  Be sure to change boolean config values to `"true"` or `"false"`, `null` to `~`, and small floats to their number version.



# Rocket-Launch

Rocket-Launch is a generalized version of the Rocket framework. Rocket-Launch can use any HuggingFace model, is capable of using any HuggingFace dataset, and utilizes PyTorch Lightning to easily enable distributed training on a number of configurations. Rocket-Launch is designed to be a flexible research framework with the ability to:

- Finetune on any dataset.
- Train from scratch on any dataset.
- Enable users to modify low-level model code and architecture
- Scale up to large models with distributed training.

Rocket-Launch primarily uses HuggingFace and PyTorch Lightning to achieve these abilities. The user is encouraged to understand these tools. In short:

- HuggingFace easily provides a wide range of models and datasets to use.
- PyTorch Lightning enables high-performance distributed training, as well as great flexibility in training code setup for a variety of needs.

This repository assumes you are running the code on a Slurm-enabled supercomputing infrastructure, but this is not necessary.

## Project Structure

This repository consists of:

- **configs**: the configuration folder holding all configs for use in training, data preparation, and evaluation.
- **dataset**: the dataset folder should store all raw and tokenized data, as well as tokenizers.
- **data_setup**: contains scripts for downloading data, most notably from the HuggingFace Hub
- **runs**: contains all results from training and evaluation jobs.
- **slurm**: slurm scripts for various tasks.
- **tokenizer**: various scripts pertaining to tokenization, as well as the core tokenizer class in [tokenizer.py](./tokenizer/tokenizer.py).
- **utils**: various utils.
- **dataset.py**: containing PyTorch Lightning DataModule class and DataSet class. These classes should be modified for specific use cases.
- **generation.py**: script for generating from trained model.
- **inference.py**: script for running inference data on given metrics or benchmarks.
- **llama.py**: core LightningModule class for Llama.
- **model.py**: model code for Llama.
- **tokenize_data.py**: tokenizes data found in corresponding path in given config.
- **train.py**: training script.

## Workflow

There are a variety of workflow approaches for a framework such as this. In general, a workflow for this repository involves:

- Downloading a dataset to a data directory.
- Training a tokenizer on the data, or using a pretrained tokenizer.
- Tokenizing the data with this tokenizer, and saving to the data directory.
- Training a model on the tokenized data.
- Running inference and/or generation with the trained model.

## Setup

### Environment

Create a Mamba environment with python=3.9, preferably named ```rocket```:
```mamba create -n rocket python=3.9```

If it is named differently, the environment activation commands in the Slurm scripts must be changed.

Run ```pip install -r requirements.txt```.

### Setting up a Config

Configuration YAML (YAML Ain't Markup Language) files are used to define all paths, settings, and hyperparameters for training tokenizers, tokenizing data, training models, and running inference on models. In the config folder, you can create a new config by copying default_config.yaml, preferebly into the [user_configs](./configs/user_configs/) folder. Fill out the class parameters accordingly.

- Any paths relating to the dataset or checkpoints should be in a directory with plenty of storage
- It's recommended to use absolute paths in the config.
- This repository is setup to work flexibly with any desired directory structure.
- This repository is setup to work flexibly with any dataset source. If retrieving datasets from the HuggingFace Hub, define the parameters to match.
- You may define paths for either one single dataset path, or seperate paths for train/test/eval dataset paths, depending on the form of the data.

### Setting up Slurm scripts

With the exception of downloading data, all steps in the pipeline are designed to be run through Slurm processes. The [slurm](./slurm/) folder contains default Slurm scripts for many steps in the pipeline. It is recommended to copy all necessary Slurm scripts into the [user_slurm](./slurm/user_slurm/) folder. Before running any Slurm script, edit the configuration to work for your usage. Ensure you are activating the right Mamba environment in the script, and that the correct config path is given.

### Getting Data

This repository can ideally be utilized with any datasource, but it is specifically setup to use datasets from HuggingFace. See [Getting Data](./docs/Getting_Data.md) for more information.

### Preparing Tokenizer

This repository is designed to work with either HuggingFace tokenizers or SentencePiece tokenizers. See the respective documentation for [HuggingFace](./docs/Training_HF_Tokenizer.md) and [SentencePiece](./docs/Training_SP_Tokenizer.md) tokenizers for more information.

### Tokenizing data

There are a number of methods for tokenizing data: it can be done in the preprocessing stage, or dynamically during training. See the docs on [tokenizing data](./docs/Tokenizing_Data.md) for more information.

## Preparing Models

This repository is designed to be work flexibly with any model architecture. By default, it uses models from the HuggingFace Transformers library. However, any PyTorch model code could be added and used with the PyTorch Lightning [Model](./src/lightning/model.py) class.

### Using HuggingFace Models

To prepare to train with a HuggingFace model, navigate to the PyTorch Lightning [model.py](./src/lightning/model.py) script. Import any necessary HuggingFace classes. Edit the `Model` class to use the proper config class, with the necessary parameters. This is highly dependent on the kind of model being used. If using a pretrained model, set the `from_pretrained` flag in the configuration YAML to `True`.

#### Modifying Model Archiecture

Many HuggingFace models encapsulate most of the necessary code into one Python file. As such, if you wish to modify architecture in any HuggingFace model, you may copy the model code from HuggingFace into a new Python file in this repository, make the necessary changes, and import that model file into the train script.

### Using PyTorch Models

Any PyTorch models can be added by simply adding a new model file, and instantiating the model object within the PyTorch Lightning [Model](./src/lightning/model.py) class.

## Training

Before training, it may be desirable change the dataset processing in [dataset.py](./dataset.py). By default, the dataset class is padding each sequence in the batch. The best processing method is highly dependent on the data.

The [train.py](./train.py) takes as an argument a path to a config yaml file. There is a slurm script, [run_train.sh](./slurm/run_train.sh) that calls this script. Edit the slurm script to use your config file, and training will begin when ran.

## Inference

There are two scripts for inference: [generation.py](./src/generation.py), for generating text, and [inference.py](./src/inference.py), for running on a test set and computing metrics.

### inference.py

For running the test set and gathering basic metrics, such as BLEU, [inference.py](./src/inference.py) can be run. To modify the metrics being gathered, modify the appropriate PyTorch Lightning hooks, such as `test_step()` and `on_test_end()`, in the [model.py](./src/lightning/model.py) script.

### generation.py

To use the model to generate text, [generation.py](./src/generation.py) can be run. Modify the `generation_path` parameter in the configuration to point to the file containing prompts to generate from.
