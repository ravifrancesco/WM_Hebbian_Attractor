# Models-of-Working-Memory

This repo contains the code necessary to reproduce Figure X and Y of the paper [Integration of recognition, episodic, and associative memories during complex human behavior](https://klab.tch.harvard.edu/publications/PDFs/gk8138.pdf). The documentation contains the necessary instructions to understand and use the classes and functions.

## Reproduce results

To reproduce the paper results:
1. Install the necessary libraries. The repo contains the `torch.yml` to create the correct conda environment by using the command `conda env create --file torch.yml`.
2. Run the `attr_behavioral_metrics.ipynb` to reproduce the results. The notebook will output the correct plots, as well as a file called `model_metrics.mat` to plot the results using matlab.
