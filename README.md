# IfA-SummerStudent-2022
This directory contains the project I worked on as a Summer Student at the Institute for Astronomy at the University of Edinburgh. The main analysis is contained in the folder `fitmodel_densitysplit/` while the remaining ones document intermediate steps I took to arrive at the final work flow.
Specifically, in `tutorials/` I gain familiarity with *nbodykit* and computing power spectra which I then utilize on an *eBOSS* mock catalog which is documented in `explore_eBOSS_powerspectrum/`. Lastly, `explore_fitting_methods/` contains some preliminary methods of fitting a model for the redshift space power spectrum to data.

All computations have been performed on the IfA's computing cluster *Cuillin* which can be accessed as described below. Afterwards I lay out the main analysis workflow which is described in detail in the project report which can also be found in ```fitmodel_densitysplit/reports```.

## Accessing Cuillin and set up
Begin by tunneling to Cuillin and logging in. 
```
username@cuillin.roe.ac.uk
```
You will need to be connected to the University's network or use their VPN service (follow the SSL VPN access section of https://www.ed.ac.uk/information-services/computing/desktop-personal/vpn/vpn-service-using)

All the packages used in this repository are included in the anconda/3.7 module. When not planning on using jupyter notebooks, the set up is completed by loading the module:
```
module load anaconda/3.7
```
When submitting a job to Cuillin, it is good practice to include this command in the bash file. An example is given in `fitmodel_densitysplit/do_fitting/submitjob.sh`

### Using Jupyther notebooks
First we need to construct a virtual environment for nbodykit. Connect to Cuillin and run
```
conda create --name nbodykit-env python=3
source activate nbodykit-env
conda install -c bccp nbodykit
```
To use the environment in a notebook a new kernel must be set up. For this, install jupyter in the new environment and add the kernel (feel free to change the name "local-venv") via:
```
pip install jupyter
ipython kernel install --name "local-venv" --user
```
See https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove for further information.
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
which can then be accessed in a browser via 
```
http://localhost:2067/tree/
```
When creating a new notebook select the "local-venv" kernel and test if the set up was successful by a simple import such as `from nbodykit.lab import cosmology`.

Once this set up has been completed, the next time you connect to Cuillin you only need to run
```
module load anaconda/3.7
source activate nbodykit-env
```
and connect to a jupyter sever as described above and select the correct kernel whenever making a new jupyter notebook.

#### Common issues when connecting to the jupyter sever
- Error stating that the adress is already in use, change the number 2067 to something else like 2066.
- Error `channel 2: open failed: connect failed: Connection refused` means that the jupyter sever has not been
  set up properly; Start a new sever, potentially using a port different from 3033. You can view the list of active severs in Cuillin by running `jupyter notebook list` after loading anaconda/3.7. To stop the server at port 3033, run `jupyter notebook stop 3033`. 

  
## Main analysis workflow

