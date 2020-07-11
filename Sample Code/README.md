# Reproducibility Guide

In this folder, we provide the codes and the link to the processed labeled dataset of this project. <br>

The data can be downloaded from this [**link**](https://drive.google.com/drive/folders/152SPVDyGgWmKyslWacTM4tLeYTRvMQPg?usp=sharing). <br> <br>


The dataset is provided in separate folders: <br>

### [Original Damage Scenarios](https://drive.google.com/drive/folders/1z12XTpZ16RhSGCv3Co-Ge8BNYXTvoXa7?usp=sharing)

<div align="justify"> In this folder, we provide data for 1,000 damage scenarios based on simulated seismic scenarios with magnitude m={6,7,8,9} (4,000 scenarios in total). The seismic scenarios are computed based on the earthquake catalog of the area encompassing Shelby Count, TN. The data is given as input to td-INDP for T=20 time-steps. We provide similar data sheets given number of resources R<sub>c</sub>={2,3,...,8}. For further details, the user is referred to the following paper. </div> <br>

[**The interdependent net-work design problem for optimal infrastructure system restoration**](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.12171?casa_token=Dx3wgv1vfkUAAAAA%3ANI2tStQRoTCrj5AmZ7LchqlvQYhmyoHHC35rgz6x39eRDvtURRIUnPeNq0uhbUxSFu-XYd06JdhDEUY). <br>

### [Augmented Scenarios (Initial Points)](https://drive.google.com/drive/folders/1ax1L9eTA0WaA-mOWe6dOAkIKEp3sj4g2?usp=sharing)

<div align="justify"> To make our model more generalizable, we augment the original damage scenarios by perturbing the original 4,000 scenarios discussed above. To this end, we choose a random number of nodes in the network and flip their binary values. Then we feed these perturbed scenarios to the simulator to obtain the labels for new training data. Consequently, we obtain 10,000 input-output data from the simulator for each earthquake magnitude ($40,000$ in total). Since the learning is more demanding for higher magnitudes of earthquake, for experimental purposes, we extend this process for m=9 (highest earthquake magnitude the simulator handles) and generate 40,000 more tailor-made scenarios for this magnitude (hence, 50,000 scenarios for m=9 and 10,000 for each of m={6,7,8}). Notes that this folder only contains the intial points (at time-step t=0) of the scenarios. Please check out [this folder](https://drive.google.com/drive/folders/1HzQ2BW7rGoIW2m0TyxUoqJ2Y87jO19Qp?usp=sharing) to obtain the simulation labeled data for T=20 time-steps. </div>
