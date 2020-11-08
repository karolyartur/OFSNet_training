# OFSNet_training
Training process for the OFSNet

This repository contains the code for fine-tuning the neural network, OFSNet (Optical Flow-based Segmentation Network)
based on the collected and prepared data. For the data collection and preparation see the [data_collection](https://github.com/karolyartur/data_collection).
The trained OFSNet is used in the [obstacle-detection](https://github.com/karolyartur/obstacle-detection/tree/deploy) repository.

## How to use
The code in this repository requires the [obstacle-detection](https://github.com/karolyartur/obstacle-detection/tree/deploy) repository to be present at the
target machine already as it contains the definitions for the OFSNet.

Clone the repository so it is in the same directory as the obstacle-detection repository:
```bash
git clone https://github.com/karolyartur/OFSNet_training
```

Make sure that the obstacle-detection repository is at the latest commit of the `deploy` branch (most up-to-date network definitions)
and that the resulted directory structure looks like this:
```bash
│  
├── obstacle-detection
├── OFSNet_training
```

After this, create a folder named `data` under `OFSNet_training` and make two subdirectories of the `data` folder called `train` and `valid`.

The resulted file structure should look like this:
```bash
│  
├── obstacle-detection
├── OFSNet_training
        │
        ├── data
        │   │
        │   ├── train
        │   ├── valid
        │
        ├── .gitignore
        ├── fine-tuning.py
        ├── README.md     
```

After preparing the datasets with the [data_collection](https://github.com/karolyartur/data_collection) repository, move the generated datasets
(the individual folders, not the whole `data`folder) into the `train` or the `valid` folders. When both folders are ready it is time to train the OFSNet.

Before running the fine-tuning of the network, make sure that the virtual environment used for the `obstacle detection` package is activated, then type:

```bash
python fine-tuning.py
```

The training of the network will start. There are three parameters that can be set by changing their value inside the python script.

 - `Batch size`: The number of inputs in a training batch (20 by default)
 - `Number of iterations`: The number of maximum iterations after which the training stops (2000 by default)
 - `Number of training steps after saving and validation`: By default after every 500 training step the model is saved and validated

## Results
After the training the results are written in a newly created folder named `fine_tuning_result` in the root of the repository.
This folder contains the fine-tuned neural network definition files and the saved training statistics.

In order to visualize the training statistics use TensorBoard with the logdir parameter set to the `fine_tuning_result` folder.

After the fine-tuning, the updated model files can be copied into the `obstacle-detection` package for use.
This way, the OFSNet definition used for the obstacle detection can be fine-tuned for new environments.