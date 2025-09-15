# Finding_Ariel
This repository contains the necessary files and some results of the project "Finding Ariel: Comparing Embedding Extractors for Zero-Shot Clustering in Danish Fish Classification"

"commands.txt" contains the organised list of commands used to achieve the results presented in the paper. The paths should be consistent, but minor modifications might be necessary. 



The respository is organised as follows: 
  - scripts folder contains all the relevant scripts to run to get the results. 
  - knn_classifier contains the results of the KNN classifier, both the best model, and csv files with the per-class perfromance, and AMI scores
  - knn_eval contains additional results of the KNN classifier, among which confusion matrices and f1 score resports 
    Both folders are to be inspected for specific results of different models
  - models_and_weigths contain both saved models and the weights used for ResNet-FishNet
  - normalised_cluster_label_distrubution contains both the division of the clusters by class labels, in the csv files, and the manual annotation of the cluster labels in the txt files 
  - images_for_paper contains the high resolution images (were possible) which were used for the paper
  
 requirements.txt contain the requirements used for this project, 
  
ChatGPT was used to generate code when necessary, to speed up the project's time. 
