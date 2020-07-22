# Deep Learning-based Resource Allocation for Infrastructure Resilience


In this repository, we provide the details of the implementation of the following manuscript: <br> <br>


### [Deep Learning-based Resource Allocation for Infrastructure Resilience](https://arxiv.org/abs/2007.05880)

Siavash Alemzadeh, Hesam Talebiyan, Shahriar Talebi, Leonardo Due&#241;as-Osorio, Mehran Mesbahi <br> <br>

## Abstract

<div align="justify"> From an optimization point of view, resource allocation is one of the cornerstones of research for addressing limiting factors commonly arising in applications such as power outages and traffic jams. In this paper, we take a data-driven approach to estimate an optimal nodal restoration sequence for immediate recovery of the infrastructure networks after natural disasters such as earthquakes. We generate data from td-INDP, a high-fidelity simulator of optimal restoration strategies for interdependent networks, and employ deep neural networks to approximate those strategies. Despite the fact that the underlying problem is NP-complete, the restoration sequences obtained by our method are observed to be nearly optimal. In addition, by training multiple models---the so-called estimators---for a variety of resource availability levels, our proposed method balances a trade-off between resource utilization and restoration time. Decision-makers can use our trained models to allocate resources more efficiently after contingencies, and in turn, improve the community resilience. Besides their predictive power, such trained estimators unravel the effect of interdependencies among different nodal functionalities in the restoration strategies. We showcase our methodology by the real-world interdependent infrastructure of Shelby County, TN. </div> <br>

<p float="left">
  &emsp;
  <img src=Figures/Map.PNG width="300" height="230" />
  &emsp; &emsp;
  <img src=Figures/Scheme.png width="450" height="230" />
</p> <br> <br>

<div align="justify"> The contributions of our work are as follows: (1) We employ neural networks and train them with high-fidelity restoration strategies devised by a mixed-integer programming formulation to predict restoration strategies in real-time, (2) we find the most efficient number of required resources based on multiple pre-trained models given the specific damage scenario, (3) we study meaningful realizations from the learned models that provide insights about restoration dynamics and the role of network interdependencies, and (4) we provide the set of labeled data that we have collected from the simulator on Shelby County, TN testbed. </div> <br>

<p float="left">
  &emsp;
  <img src=Figures/Matrix.png width="310" height="286" />
  &emsp; &emsp;
  <img src=Figures/NN.png width="419" height="286" />
</p> <br> <br>
