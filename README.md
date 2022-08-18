# IfA-SummerStudent-2022
This directory contains the project I worked on as a Summer Student at the Institute for Astronomy at the University of Edinburgh. The main aim of the project is to fit a simple model (Kaiser model with Finger of God term hereafter referred to as KaiserFoG model) for the redshift space power spectrum to BGS mock data. The fitting will be performed several times, each time using galaxies from a different density interval yielding tracer fields with varying biases but sampling the same volume of space. The cross-correlation between such fields is predicted to contain relativistic effects which we hope to detect in DESI's BGS data [[Beutler et al.]](https://doi.org/10.48550/arXiv.2004.08014). The purpose of this project is to assess on what scales the KaiserFoG model provides accurate enough predictions to be used in the BGS analysis. Therefore, the fitting is performed for a selection of $k$ ranges.

The main analysis is contained in the folder `fitmodel_densitysplit/` and documented in detail in project report which can be found alongside a presentation and poster in the folder `documentation/`. The main steps of the analysis workflow are laid out below.
The remaining folders of this directory contain intermediate steps I took to arrive at the final workflow. Specifically, in `tutorials/` I gain familiarity with [nbodykit](https://nbodykit.readthedocs.io/en/latest/) and computing power spectra which I then apply to an *eBOSS* mock catalogue, documented in `explore_eBOSS_powerspectrum/`. Lastly, `explore_fitting_methods/` investigates different methods of fitting model parameters.

All computations have been performed on the IfA's computing cluster *Cuillin* which can be accessed as described below. Note that the BGS data used for the fitting is stored on Cuillin and requires read permissions.


## Main analysis workflow
Here I provide an overview of what tasks are performed by which files (with paths relative to `fitmodel_densitysplit/`). Please consider the project report for a more in depth discussion.

1. Compute and store the 2D redshift space power spectrum for each density bin: `getpower_densitybins/`. A visualisation of the density partition is provided by `show_densityfield.ipynb`.
2. Estimate the covariance matrix of the monopole and quadrupole for each density bin: `bruteforce_covmat/`
3. Perform a fast estimation of the KaiserFoG model parameter values and store these: `pre-analysis_fitting.ipynb`
4. Fit the power spectrum model in each density bin using Markov Chain Monte Carlo, store the results, and make plots illustrating the quality of the fit: `do_fitting/`. This fitting is performed for an incrementally enlarged range of $k$ and the fitted model parameters are plotted as a function of $k_{max}$, the upper bound of the consider $k$ range.
5. Compare the data and the predictions by the pure Kaiser and KaiserFoG models: `compare_data_fittedmodels.ipynb`

The computationally expensive steps 1, 2, and 4 are designed to be submitted as jobs to Cuillin. Information on how to do so can be found [here](https://cuillin.roe.ac.uk/projects/documentation/wiki/Job_Submission). Alternatively, one may use the bash files in the directories of the referenced steps as templates. 


## Accessing Cuillin and setup
Begin by tunnelling to Cuillin and logging in. 
```
username@cuillin.roe.ac.uk
```
You will need to be connected to the University's network or use their VPN service (follow the SSL VPN access section [here](https://www.ed.ac.uk/information-services/computing/desktop-personal/vpn/vpn-service-using))

All the packages used in this repository are included in the anaconda/3.7 module. When not planning on using jupyter notebooks, the setup is completed by loading the module:
```
module load anaconda/3.7
```
When submitting a job to Cuillin, it is good practice to include this command in the bash file. An example is given in `fitmodel_densitysplit/do_fitting/submitjob.sh`.

### Using jupyter notebooks
First, you need to construct a virtual environment for nbodykit. Connect to Cuillin and run
```
conda create --name nbodykit-env python=3
source activate nbodykit-env
conda install -c bccp nbodykit
```
To use the environment in a notebook a new kernel must be set up. For this, install jupyter in the new environment and add a kernel via:
```
pip install jupyter
ipython kernel install --name "nbodykit-env" --user
```
Further information on jupyter kernels can be found [here](https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove).
Next, start a new jupyter sever:
```
screen -S jupyter_server
module load anaconda/3.7
jupyter notebook --no-browser --port=3033
Ctr+a+d
``` 
To connect to it, open a new terminal and run
```
ssh -N -f -L localhost:2067:localhost:3033 username@cuillin.roe.ac.uk
```
which can be accessed via a browser at `http://localhost:2067/tree/`.
When creating a new notebook select the "nbodykit-env" kernel and test if the set up was successful by a simple import such as `from nbodykit.lab import cosmology`.

Once this setup has been completed, the next time you connect to Cuillin you only need to run
```
module load anaconda/3.7
source activate nbodykit-env
```
and connect to a jupyter sever as described above and select the correct kernel whenever making a new jupyter notebook.

#### Common issues when connecting to the jupyter sever
- Error stating that the address is already in use: Change the address 2067 to something else like 2066.
- Error `channel 2: open failed: connect failed: Connection refused`: The jupyter sever has not been set up properly; Start a new sever, potentially using a port different from 3033. You can view the list of active severs by running `jupyter notebook list` after loading anaconda/3.7. To stop the server at port 3033, run `jupyter notebook stop 3033`.
