# Cross Network Source Localization
The goal of this project is to source localization of cross network.
<!---
## Sample Case
In Stackoveflow people share Github repositories. 
A malicious Github repositories can affect other repositories in Github as well as Stackoverflow.
For those two separate networks, if we know the affected posts of Stackoverflow, can we localize the source repositories on Github?
--->
## Problem Formulation

![Cross Network Influence](./img/crossNetCase.png)

Two networks, nodes and edges are: 
* Projection Net, 𝐺<sub>𝑝𝑟𝑜𝑗</sub>, 𝑉<sub>𝑝𝑟𝑜𝑗</sub>, 𝐸<sub>_proj_</sub>
* Receiving Net, 𝐺<sub>𝑟𝑒𝑐</sub>, 𝑉<sub>𝑟𝑒𝑐</sub>, 𝐸<sub>𝑟𝑒𝑐</sub>

Edges:
* Observed Edges (Black), 𝐸<sub>𝑝𝑟𝑜𝑗</sub>, 𝐸<sub>𝑟𝑒𝑐</sub>
* Diffusion Edges
    * Projection Net (Purple), 𝐸'<sub>𝑝𝑟𝑜𝑗</sub>
    * Receiving Net (Blue), 𝐸'<sub>𝑟𝑒𝑐</sub>
    * Cross Net (Green Dotted), 𝐸<sub>𝑐𝑟𝑜𝑠𝑠</sub>
    
Example: 
* Yellow: 𝑣<sub>𝑝𝑟𝑜𝑗</sub><sup>𝑠</sup> = 3 started in 𝐺<sub>𝑝𝑟𝑜𝑗</sub> which infected 𝑉<sub>𝑟𝑒𝑐</sub><sup>i</sup>= [2, 8, 11, 12] in 𝐺<sub>𝑟𝑒𝑐</sub>
* Orange: 𝑣<sub>𝑝𝑟𝑜𝑗</sub><sup>𝑠</sup> = 4 started in 𝐺<sub>𝑝𝑟𝑜𝑗</sub> which infected 𝑉<sub>𝑟𝑒𝑐</sub><sup>i</sup>  = [5, 9] in 𝐺<sub>𝑟𝑒𝑐</sub>

Goal:

Find the nodes in 𝐺_𝑝𝑟𝑜𝑗  (Here 3 and 4) which are responsible for the infection in 𝐺<sub>𝑟𝑒𝑐</sub> (Here [2, 8, 11, 12] and [5, 9]).
𝑉<sub>𝑟𝑒𝑐</sub><sup>i</sup>  →𝑉<sub>𝑝𝑟𝑜𝑗</sub><sup>𝑠</sup>
{[2, 8, 11, 12], [5, 9]} -> [3, 4]

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

### Plotting
Look at the [plotting](./plotting) folder for the plotting of different stages.

[plotting_vae.py](./plotting/plotting_vae.py) will show the initial training of VAE:
![Training_VAE](./saved_models/VAE_train_log_github2stackTest__LT2IC102023-11-10%2016-03-47.png)

[plotting_diffProj.py](./plotting/plotting_diffProj.py) will show the initial training of diffusion model for first network:
![Training_Diff_Proj](./saved_models/diffProj_train_log_github2stackTest__LT2IC102023-11-10%2016-04-01.png)

[plotting_diffRec.py](./plotting/plotting_diffRec.py) will show the initial training of diffusion model for second network:
![Training_Diff_Rec](./saved_models/diffRec_train_log_github2stackTest__LT2IC102023-11-10%2016-04-18.png)

[plotting_all.py](./plotting/plotting_all.py) will show the final training of the whole framework:
![Training_All](./saved_models/train_log_github2stackTest__LT2IC102023-11-10%2016-04-45.png)

[plotting_all.py](./plotting/plotting_inference.py) will show the progress of inference:
![Inference](./saved_models/Inference_log_github2stackTest__LT2IC102023-11-10%2018-39-14.png)
