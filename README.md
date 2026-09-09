# Analysis and modelling of Hantavirus outbreaks
This repository contains the code and material to reproduce the results of the paper "Models and preparedness scenarios for horizontal transmission in Andes Hantavirus outbreaks", for the analysis, uncertainty quantification and scenario modelling of past and recent Andes virus (ANDV) outbreaks. 

## Folders and files
Two main Jupiter notebooks contain the whole analysis:
- **Hanta_analysis.ipynb** is for analysing of the outbreaks, fitting epidemiological parameters, getting started with toy scenarios and perform the meta-analysis on $R_0$.
- **Hanta_scenarios.ipynb** is for developing the uncertainty analysis on epidemiological and network parameters, assessing the impact of superspreading and developing containment scenarios.

Both nothebooks are subdivided into sections, dedicated to each of the developed tasks. They fully return the analysis and figures reported in the article.  
Hanta.scenarios.ipynb saves checkpoint simulation results in a folder "data" to speed up computation. The first time the notebook is run, make sure that the folder is available and the related code is decommented.  
Both notebooks save figures (when the "save" line is decommented) in a folder "figures".  
Both notebooks rely on custom functions, located in the "src" folder.

### src
Custom functions for simulations and plotting are contained here. The file names point to functions dedicated to fitting different parameters, simulating the network model, performing uncertainty analysis, or performing sensitivity analysis ("sweep") over multiple parameters of interest.  
The comments to these files have been generated using Claude Sonnet 4.6.

## Requirements
Reproducng the analysis requires Jupyter notebook (tested on v. 7.6.2) and Python 3.  
The code uses basic python functions (scipy, numpy), plus custom functions located in "src". Generating the scale-free network requires the library `networkx`.  
Library versions are as follows:  
- scipy==1.17.0
- pandas==2.3.3
- numpy==2.4.6
- networkx==3.6.1
- matplotlib==3.10.9

## Reproducing the results
Run (`Run All Cells`) the whole notebook `Hanta_analysis.ipynb` to obtain the analysis of data, estimation of parameters and related figures.  
Run (`Run All Cells`) the whole notebook `Hanta_scenarios.ipynb` to obtain the analysis of scenarios, effect of network structures, parameters and other conditions.  

Running both notebooks does not require specific inputs (data are already coded within the notebooks) and delivers the statistical results and figures included in the paper. Both notebooks typically run in a few minutes on an average laptop. The time may extend significantly depending on the considered network size used for Bayesian fitting or simulations, as well as on the number of repeated runs. With default settings, `Hanta_analysis.ipynb` requires about 15 minutes, while `Hanta_scenarios.ipynb` is completed in about 5 minutes.

## Credits
Code developed by Daniele Proverbio.  
If you wish to reuse the code, please cite its companion article: Proverbio and Giordano, "_Andes_ Hantavirus human-to-human transmission dynamics and
preparedness scenarios", 2026
