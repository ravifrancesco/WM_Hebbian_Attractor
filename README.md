# Models-of-Working-Memory

This repository contains the code necessary to reproduce Figure X and Y of the paper [Integration of recognition, episodic, and associative memories during complex human behavior](https://www.biorxiv.org/content/10.1101/2023.03.27.534384v1). The documentation contains the necessary instructions to understand and use the classes and functions.

## Reproduce results

To reproduce the paper results:
1. Install the necessary libraries. The repository contains the `torch.yml` to create the correct conda environment (use the command `conda env create --file torch.yml`).
2. Run the `attr_behavioral_metrics.ipynb` to reproduce the results. The notebook will output the correct plots, as well as a file called `model_metrics.mat` to plot the results using matlab.
