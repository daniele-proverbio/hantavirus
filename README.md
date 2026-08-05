# Analysis and modelling of Hantavirus outbreaks
This repository contains the code and material to reproduce the results of the paper "Models and preparedness scenarios for horizontal transmission in Andes Hantavirus outbreaks", for the analysis, uncertainty quantification and scenario modelling of past and recent Andes virus (ANDV) outbreaks. 

## Folders and files
Two main Jupiter notebooks contain the whole analysis:
- **Hanta_analysis.ipynb** is for analysing of the outbreaks, fitting epidemiological parameters, getting started with toy scenarios and perform the meta-analysis on $R_0$.
- **Hanta_scenarios-ipynb** is for developing the uncertainty analysis on epidemiological and network parameters, assessing the impact of superspreading and developing containment scenarios.

Both nothebooks are subdivided into sections, dedicated to each of the developed tasks. They fully return the analysis and figures reported in the article.  
Hanta.scenarios.ipynb saves checkpoint simulation results in a folder "data" to speed up computation. The first time the notebook is run, make sure that the folder is available and the related code is decommented.  
Both notebooks save figures (when the "save" line is decommented) in a folder "figures".  
Both notebooks rely on custom functions, located in the "src" folder.

### src
Custom functions for simulations and plotting are contained here. The file names point to functions dedicated to fitting different parameters, simulating the network model, performing uncertainty analysis, or performing sensitivity analysis ("sweep") over multiple parameters of interest.  
The comments to these files have been generated using Claude Sonnet 4.6.

## Requirements
Basic python functions (scipy, numpy), plus custom functions located in "src".

## Credits
Code developed by Daniele Proverbio.  
If you wish to reuse the code, please cite its companion article: Proverbio and Giordano, "Models and preparedness scenarios for horizontal transmission in Andes Hantavirus outbreaks", 2026
