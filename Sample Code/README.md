# Reproducibility Guide

In this folder, we provide the codes and the link to the processed labeled dataset of this project. <br>

The data can be downloaded from this [**link**](https://drive.google.com/drive/folders/152SPVDyGgWmKyslWacTM4tLeYTRvMQPg?usp=sharing). <br>

---

The dataset is provided in separate folders: <br>

### [Original Damage Scenarios](https://drive.google.com/drive/folders/1z12XTpZ16RhSGCv3Co-Ge8BNYXTvoXa7?usp=sharing)

<div align="justify"> In this folder, we provide data for 1,000 damage scenarios based on simulated seismic scenarios with magnitude m={6,7,8,9} (4,000 scenarios in total). The seismic scenarios are computed based on the earthquake catalog of the area encompassing Shelby Count, TN. The data is given as input to td-INDP for T=20 time-steps. We provide similar data sheets given number of resources R<sub>c</sub>={2,3,...,8}. For further details, the user is referred to the following paper. </div> <br>

[**Link to the paper**](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.12171?casa_token=Dx3wgv1vfkUAAAAA%3ANI2tStQRoTCrj5AmZ7LchqlvQYhmyoHHC35rgz6x39eRDvtURRIUnPeNq0uhbUxSFu-XYd06JdhDEUY) <br>

---

### [Augmented Damage Scenarios](https://drive.google.com/drive/folders/1ax1L9eTA0WaA-mOWe6dOAkIKEp3sj4g2?usp=sharing)

<div align="justify"> To make our model more generalizable, we augment the original damage scenarios by perturbing the original 4,000 scenarios discussed above. To this end, we choose a random number of nodes in the network and flip their binary values. Then we feed these perturbed scenarios to the simulator to obtain the labels for new training data. Consequently, we obtain 10,000 input-output data from the simulator for each earthquake magnitude (40,000 in total). Since the learning is more demanding for higher magnitudes of earthquake, for experimental purposes, we extend this process for m=9 (highest earthquake magnitude that td-INDP can handle) and generate 40,000 more tailor-made scenarios (hence, 50,000 scenarios for m=9 and 10,000 for each of m={6,7,8}). Notes that this folder only contains the intial points (i.e., recovery state vector at time-step t=0) of the scenarios. Please check out the following folder to obtain the simulation labeled data for T=20 time-steps. </div> <br>
  
 [**Link to the folder**](https://drive.google.com/drive/folders/1HzQ2BW7rGoIW2m0TyxUoqJ2Y87jO19Qp?usp=sharing) <br>
 
 ---

**How to interpret the data sheets:** <br>

<div align="justify"> Each processed labeled data is provided in three .csv files that contain the recovery time-series data of 49 water, 16 gas, and 60 power nodes respectively. The information in the tables are in the form of the figure below: </div> <br>

<img src=../Figures/Table_guide.PNG width="1000" height="466" /> <br>

<div align="justify"> Each row represents the recovery state of one node in T=20 steps. For instance, in the figure above, node 12 is damaged (value equal to zero) in the beginning of the recovery process and gets fixed at time-step t=4. Each column denotes the time-step. The goal of our method approximates the pattern of recovery shown the figure above. We use the augmented scenarios as our training set and the original scenarios as the test set. We then evaluate the accuracy of our trained model by calculating an index measure called Accuracy Radius **AR**, which is defined as the acceptable margin of prediction provided by the trained neural net. The results provided below are discussed in detail in the manuscript. </div> <br>

<img src=../Figures/accuracyVsMag.PNG width="600" height="400" /> <br>
