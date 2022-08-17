# IfA-SummerStudent-2022
This directory contains the project I worked on as a Summer Student at the Institute for Astronomy at the University of Edinburgh. The main analysis is contained in the folder `fitmodel_densitysplit/` while the remaining ones document intermediate steps I took to arrive at the final work flow.
Specifically, in `tutorials/` I gain familiarity with *nbodykit* and computing power spectra which I then utilize on an *eBOSS* mock catalog which is documented in `explore_eBOSS_powerspectrum/`. Lastly, `explore_fitting_methods/` contains some preliminary methods of fitting a model for the redshift space power spectrum to data.

All computations have been performed on the IfA's computing cluster *Cuillin* which can be accessed as described below. Afterwards I lay out the main analysis workflow which is described in detail in the project report which can also be found in ```fitmodel_densitysplit/reports```.

## Accessing Cuillin
- To tunnle to Cuillin, you need to be connected to the University's network or use their VPN (follow the SSL VPN access section of https://www.ed.ac.uk/information-services/computing/desktop-personal/vpn/vpn-service-using)
- Once connected, tunnel to Cuillin: 
```
username@cuillin.roe.ac.uk
```
  - To use jupyter notebooks, start a new jupyter sever. This needs to be only done once:
  ```
    screen -S jupyter_server
	 module load anaconda/3
	 jupyter notebook --no-browser --port=3033
	 Ctr+a+d
  ``` 
  - To tunnel to the jupyter sever open a new terminal and run
  ```
  ssh -N -f -L localhost:2067:localhost:3033 username@cuillin.roe.ac.uk
  ```
  which can then be accessed in a browser via ```http://localhost:2067/tree/```
  Should this yield the error `channel 2: open failed: connect failed: Connection refused`, then the jupyter sever has not been
  set up properly; Start a new sever, potentially using a port different from 3033.
- All packages used in the project are avaialbale in the anaconda/3.7 module. Once connected to *Cuillin*, complete the set up by running
```
module load anaconda/3.7
```
  - Rather than using *nbodykit* though the anaconda module, one can also create a virtual environment via
```
conda create --name nbodykit-env python=3
source activate nbodykit-env
conda install -c bccp nbodykit
```
To enble to environment run `source activate nbodykit-env` and to use it jupyter notebooks, a new kernel must be constructed which is detailed here: https://queirozf.com/entries/jupyter-kernels-how-to-add-change-remove
  
## Main analysis workflow

