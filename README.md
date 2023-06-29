# Bayesian-Perturbation

The folder "Main Experiments" includes all the code for the numerical experiments of the main manuscript. 

The experiments are conducted on two different regulatory networks, which are separated in two folders: "Mammalian Cell Cycle Network" and "Gut Microbial Community Network". 

Different test cases are considered for both networks; these test cases are separated by different folders. The name of the test case corresponds to the number and type of the assumed unknown interactions.

Each test case contains "Train" and "Tests" folders. The "Train" folder corresponds to the training of our proposed policy for that specific case, and the "Tests" folder contains different analyses and performance comparisons using our trained proposed policy, *Expected Information Gain (EIG, or AL)* policy, *Maximum aposteriori (MAP)* policy, and *No Perturbation* policy.
