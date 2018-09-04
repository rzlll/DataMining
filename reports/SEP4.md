##### Update:
Last week, I included **three** more related algorithms with proper implementation (last three in the list), and now we have **seven** related works in total. 
Here's the list of the algorithms. We also updated 

1. `rev2`: Srijan Kumar, et al. REV2: Fraudulent User Prediction in Rating Platforms. (WSDM 2016)
+ `bad`: Abhinav Mishra, Arnab Bhattacharya. Finding the Bias and Prestige of Nodes in Networks based on Trust Scores. (WWW 2011)
+ `bn`: Bryan Hooi, et al. Bayesian Inference for Ratings-Fraud Detection. (SIAM 2016)
+ `trust`: Guan Wang, Sihong Xie, Bing Liu, Philip Yu. Review graph based online store review spammer detection. (ICDM 2011)
+ `feagle`: Leman Akoglu et al. Opinion Fraud Detection in Online Reviews by Network Effects. (AAAI 2013)
+ `fraudar`: Bryan Hooi et al. Fraudar: Bounding Graph Fraud in the Face of Camouflage. (KDD 2016)
+ `rsd`: Guan Wang et al. Review Graph Based Online Store Review Spammer Detection. (ICDM 2011)


##### Progressing:
Algorithms (2,3,4,6, and 7) on the *ALPHA* data are almost finished.
However, `rev2` and `feagle` still need sometime to finish on *ALPHA* data.
Algorithms (2, 3, 4) on *OTC* data are also finished.
All the other algorithms on *AMAZON* and *EPINIONS* are remaining to be done.

##### Evaluation:
We use **precision, recall, f1 score** at top @q to measure the attacks for different algorithm/data pair.

The `rev2` algorithms performed best in terms of recall and f1 score, but it dropped down once sockpuppets start to work.
The `rev2` only finishes a part of parameters.
It may perform better after all the parameters are done.
The `trust` seemed to be the most robust against this kind of attacks.


##### The next step:
The *ALPHA* and *OTC* data are smaller than *AMAZON* and *EPINIONS* data.
The immediate next step is to test the attack against the two bigger datasets (*AMAZON* and *EPINIONS*).
The other thing is to come up with ideas to defend this kind of attack.
The `birdnest` and `rev2` algorithm took look at the distributions of the rating.
However, the `trust` and `rsd`, which had better performance against the attack, focused on the reliability/trustness of the user.
It seems we should focus on the user side more than the ratings.
