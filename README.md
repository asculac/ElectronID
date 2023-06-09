# EgammaMVA
The tools to train electron and photon ID.

## Requirements

This toolkit requires the __xgbo package__ to be installed:
<https://www.github.com/guitargeek/xgbo>

## How to train an Electron MVA ID

### Overview

The procedure splits up in a few fundamental steps:

1. Make training ntuple with CMSSW.
2. Train the ID with XGBoost.
3. Determine working points.
4. Generate configuration files to integrate ID into CMSSW.
5. Make validation ntuple with CMSSW.
6. Draw performance plots and generate summary slides.

Only step 1 and 4 require interaction with CMSSW, the other steps can be done offline.

### Step 0 - Clone this repository and tweak the configuration

Adapt the configuration in `config.py` to your needs.

### Step 1 - Make training Ntuples

Start by setting up the CMSSW area e.g. CMSSW_10_6_20:

* `cmsrel CMSSW_10_6_20`
* `cd CMSSW_10_6_20/src`
* `cmsenv`

Checkout the needed packages:

* `git cms-addpkg RecoEgamma/ElectronIdentification`

Make sure you have crab in your environment and launch the job to ntuplize the sample you configured for the training:

* `python submit_ntuplizer.py --train`

When the job is done, you should merge the crab output files to one nice root file. So far, this is adapted to the environment in the LLR institute, but more general versions of this script which can be called from lxplus should be provided in the future:

* `python merge_ntuple.py --train`

### Step 2 - Train the ID with XGBoost

Launch the training with:

* `python training.py`

The code to train with TMVA can be found in `legacy_training.py`.

### Step 3 - Determine the working points

Working pointns are derived based on how you configured them. Different types of working points are supported, right now a flat cut targeting a specific signal efficiency in each training bin, and an exponential cut which fits a pt-differential signal efficiency specified in a text file.

* `python find_working_points.py`

### Step 4 - CMSSW configuration

To generate cff files for VID and save them im a CMSSW-like directory structure, together with the required weight files:

* `python make_cmssw_config.py`

Now you should probably create a new branch in your CMSSW, because next we will copy this directory structure we just created into the real cmsssw:

* `rsync -avz --verbose cmssw/<tag of your submit version>/src $CMSSW_BASE`

Note that this also updates the cfg file for the ElectronMVA Ntuplizer, which is now loaded with your new ID.

### Step 5 - Make validation ntuple with CMSSW

As we now implemented the ID in CMSSW, it is time to launch a new job to create the testing ntuple:

* `python submit_ntuplizer.py --test`

Again, don't forget to merge the ROOT files:

* `python merge_ntuple.py --test`

### Step 6 - Draw performance plots and generate summary slides

You can generate the ROC, pt, eta and pilup curves with this script:

* `python make_plots.py`
