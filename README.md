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
![Training_All](./saved_models/train_log_github2stackTest__LT2IC102023-11-10%2016-04-45.png)

[plotting_inference.py](./plotting/plotting_inference.py) will show the progress of inference:
![Inference](./saved_models/Inference_log_github2stackTest__LT2IC102023-11-10%2018-39-14.png)
