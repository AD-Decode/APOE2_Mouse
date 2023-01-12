# Folders:

## data
The RDA files include connectomes, traits, and RNA-seq. Can be loaded from Rstudio console or script.

## multi_cca
multi_cca codes: in each folder apoe2_mikes_sig_age, apoe2_mikes_sig_sex there's a multicca_age.R that can be run after loading connectivity, response and RNA_data of the root folder. The mikes_age.csv and mikes_sex.csv used in this code are FC of differential gene expression to only pick FDR significant of each age, or sex when running SMCCA.  bootcca_multi_cca.R in each folder can be run right after running multicca_age.R to compute bootstrap confidence intervals for the sum of 3 correlations. 

## rna_seq
codes and data to generate normalized counts, differential gene expression and add symbols to the gene last column. 

## vertex
Code to run vertex screening on python (vertex_func.py in the root first must be run on python console such as spider) as well as testing the result for k-fold Neural network classifiers (FNN, CGN). age_cat folder contains vertex_connectome_agecat.py that uses response.rda in this folder and connectivity.rda file in the root vertex folder to run the vertex screening python function on age category as response value. filtering.R would filter the whole connectome based on the result of this process report_age_cat.csv and generate image_filter.rda and image_all.rda.  Then kfold_no_ssir.ipynb runs the Forward neural network on the results without considering the screening and kfold.ipynb with considering the screening (each of these two files first start with logistic and end up with another run of multi-perceptron NN). NN_sex folder does a similar thing to the age_cat folder but for sex as response value y in vertex screening- the sex vertex screening file is in the vertex folder named vertex_connectome.py. GNN folder is a similar process but with Graph neural network. 


## volcano plot
Volcano plots for sex specific mike_sex.R and for mike_age.R for age specific. multi_cca_volcano is similar to #multi_cca folder but it applies the filter of smcca to FDR significant rna seqs before creating volcano plot for example via mike_age_volcano.R.  


## winding_multi_cca
MCCA of connectomes, winding numbers, RNA-seq plus anova and posthoc stats associated with AWN. multicca.R and bootcca_multi_cca.R take care of SMCCA by the files contained in this folder for input and anova_winding_apoe2.R take care of the AWN stats and plots.  

## CHASSSYMMETRIC2Legends0907...
The Atlas details used for computations and analysis.
