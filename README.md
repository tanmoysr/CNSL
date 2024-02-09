# Cross Network Source Localization

## Project Setup
Install the required packages.

`pip install -r requirements.txt`

## Running Project
### Training
For running the default training:

`python train_model.py`

If you want to change the configuration, you can do it in the file [configuration.py](./main/configuration.py) or you can follow the following example and change the parameter as per your needs.

* For example to change the number of epochs for training:

`python train_model.py -e 100`

### Inference
For running the default inference:

`python run_inference.py`

If you want to change the configuration, you can do it in the file [configuration.py](./main/configuration.py) or you can follow the following example and change the parameter as per your needs.

* For example to change the number of epochs for inference:

`python run_inference.py -eInfer 100`
