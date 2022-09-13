# Preference Selection Automatic Cut


## Introduction

The objective of this project is to integrate automatic dendrogram cut methods into the preference based model fitting algorithms T-Linkage and Mutlilink.
Our integration works by modifying the original algorithms in order to complete the preference dendrogram which would be left incomplete in the original iterations. 
We then performed fine tuning on the model parameters in order to optimize the performance, finally we tested the results of our project on plane and motions segmentation on the Adelaide dataset.

For more details you can refer to our project's documentation at https://github.com/AlessandroMessori/Preference-Selection-Automatic-Cut/blob/main/Documentation.pdf


## Results
Testing on the Adelaide dataset, we were able to reduce the Mean Percentare Error of plane segmentation with T-Linkage of 7%

## References

The automatic cut methods used in this project are taken from the following sources:

https://gmarti.gitlab.io/ml/2017/05/12/cut-a-dendrogram.html

https://towardsdatascience.com/automatic-dendrogram-cut-e019202e59a7
