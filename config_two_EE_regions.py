###########
# 2 0 1 6 #
###########

import os

if 'CMSSW_BASE' in os.environ:
    cmssw_base = os.environ['CMSSW_BASE']
else:
    cmssw_base = ''

cfg = {}

cfg['ntuplizer_cfg'] = cmssw_base + '/src/RecoEgamma/ElectronIdentification/python/Training/ElectronMVANtuplizer_cfg.py'

cfg['storage_site'] = 'T2_FR_GRIF_LLR'
cfg["submit_version"] = "DY_2016_5_12_2018"
# Location of CRAB output files
cfg["crab_output_dir"] = '/store/user/mkovac/Egamma/%s' % cfg["submit_version"]
cfg["crab_output_dir_full"] = "/dpm/in2p3.fr/home/cms/trivcat%s" % cfg["crab_output_dir"]
# Where to store the ntuples and dmatrices
cfg["ntuple_dir"] = "/home/llr/cms/kovac/CMS/RUN_2/Data/EGamma/"
cfg['dmatrix_dir'] = "/home/llr/cms/kovac/CMS/RUN_2/Data/EGamma/"
cfg['out_dir'] = "out"
cfg['cmssw_dir'] = "cmssw"

# The sample used for training and evaluating the xgboost classifier.
cfg["train_eval_sample"] = '/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/MINIAODSIM'
cfg["train_eval_sample_request_name"] = 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-train'

# The fraction of this sample used for training.
cfg["train_size"] = 0.75

# The sample used for unbiased testing (performance plots).
cfg["test_sample"] = '/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext2-v1/MINIAODSIM'
cfg["test_sample_request_name"] = 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8-test'

cfg["selection_base"] = "genNpu > 1"
cfg["selection_sig"]  = "matchedToGenEle == 1"
cfg["selection_bkg"]  = "matchedToGenEle == 0 | matchedToGenEle == 3"
#cfg["selection_id"]   = "Spring16HZZV1wpLoose > 0"

###########
# Variables
###########

variables_base = [
             "ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi",
             "ele_oldcircularity", "ele_oldr9", "ele_scletawidth",
             "ele_sclphiwidth", "ele_oldhe", "ele_kfhits", "ele_kfchi2",
             "ele_gsfchi2", "ele_fbrem", "ele_gsfhits",
             "ele_expected_inner_hits", "ele_conversionVertexFitProbability",
             "ele_ep", "ele_eelepout", "ele_IoEmIop", "ele_deltaetain",
             "ele_deltaphiin", "ele_deltaetaseed", "rho",
            ]

variables_ee_only = ["ele_psEoverEraw"]

variables_iso_only = ["ele_pfPhotonIso", "ele_pfChargedHadIso", "ele_pfNeutralHadIso"]

variables_noiso_eb = variables_base
variables_noiso_ee = variables_base + variables_ee_only
variables_iso_eb   = variables_noiso_eb + variables_iso_only
variables_iso_ee   = variables_noiso_ee + variables_iso_only


###############################
# Configuring the training bins
###############################

# Configure the different trainings.
# For each bin, you have:
#     - cut
#     - list of variables to use

cfg["trainings"] = {}
cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"] = {}

# Iso ID
cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EB1_5"] = {
        "cut": "ele_pt < 10. & abs(scl_eta) < 0.800",
        "variables": variables_iso_eb,
        "label": r'EB1 5 - 5 < $p_T$ < 10 GeV, ($|\eta| < 0.8$)',
        }

cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EB2_5"] = {
        "cut": "ele_pt < 10. & abs(scl_eta) >= 0.800 & abs(scl_eta) < 1.479",
        "variables": variables_iso_eb,
        "label": r'EB2 5 - 5 < $p_T$ < 10 GeV, ($|\eta| > 0.8$)',
        }

cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EB1_10"] = {
        "cut": "ele_pt >= 10. & abs(scl_eta) < 0.800",
        "variables": variables_iso_eb,
        "label": r'EB1 10 - $p_T$ > 10 GeV, ($|\eta| < 0.8$)',
        }

cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EB2_10"] = {
        "cut": "ele_pt >= 10. & abs(scl_eta) >= 0.800 & abs(scl_eta) < 1.479",
        "variables": variables_iso_eb,
        "label": r'EB2 10 - $p_T$ > 10 GeV, ($|\eta| > 0.8$)',
        }

cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EE1_5"] = {
        "cut": "ele_pt < 10. & abs(scl_eta) >= 1.479 & abs(scl_eta) < 2.0",
        "variables": variables_iso_ee,
        "label": r'EE1 5 - 5 < $p_T$ < 10 GeV',
        }
        
cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EE2_5"] = {
        "cut": "ele_pt < 10. & abs(scl_eta) >= 2.0",
        "variables": variables_iso_ee,
        "label": r'EE2 5 - 5 < $p_T$ < 10 GeV',
        }
        
cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EE1_10"] = {
        "cut": "ele_pt >= 10. & abs(scl_eta) >= 1.479 & abs(scl_eta) < 2.0",
        "variables": variables_iso_ee,
        "label": r'EE1 10 - $p_T$ > 10 GeV',
        }
        
cfg["trainings"]["Spring_16_ID_ISO_two_EE_regions"]["EE2_10"] = {
        "cut": "ele_pt >= 10. & abs(scl_eta) >= 2.0",
        "variables": variables_iso_ee,
        "label": r'EE2 10 - $p_T$ > 10 GeV',
        }


################################
# Configuring the working points
################################

#import numpy as np
#wp90_target = np.loadtxt("wp90.txt", skiprows=1)
#wp80_target = np.loadtxt("wp80.txt", skiprows=1)
#
#pt_bins     = wp90_target[:,:2]
#wp90_target = wp90_target[:,2]
#wp80_target = wp80_target[:,2]

#pt_bins     = []
#wp90_target = []
#wp80_target = []

cfg["working_points"] = {}
cfg["working_points"]["Summer_16_ID_ISO"] = {}

#cfg["working_points"]["Fall17NoIsoV2"]["mvaEleID-Fall17-noIso-V2-wp90"] = {
#        "categories": ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
#        "type": "pt_scaling_cut_sig_eff_targets",
#        "ptbins": pt_bins,
#        "targets": [wp90_target]*6,
#        "match_boundary": True
#        }
#
#cfg["working_points"]["Fall17NoIsoV2"]["mvaEleID-Fall17-noIso-V2-wp80"] = {
#        "categories": ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
#        "type": "pt_scaling_cut_sig_eff_targets",
#        "ptbins": pt_bins,
#        "targets": [wp80_target]*6,
#        "match_boundary": True
#        }
#
#cfg["working_points"]["Fall17NoIsoV2"]["mvaEleID-Fall17-noIso-V2-wpLoose"] = {
#        "type": "constant_cut_sig_eff_targets",
#        "categories": ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
#        "targets": [0.9475128258132411, 0.9490611850552356, 0.9044423070825363, 0.9948310845914105, 0.9914069066168602, 0.9839520294782502], # From Spring16HZZ
#        "match_boundary": True
#        }
#
#cfg["working_points"]["Fall17IsoV2"]["mvaEleID-Fall17-iso-V2-wp90"] = {
#        "categories": ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
#        "type": "pt_scaling_cut_sig_eff_targets",
#        "ptbins": pt_bins,
#        "targets": [wp90_target]*6,
#        "match_boundary": True
#        }
#
#cfg["working_points"]["Fall17IsoV2"]["mvaEleID-Fall17-iso-V2-wp80"] = {
#        "categories": ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
#        "type": "pt_scaling_cut_sig_eff_targets",
#        "ptbins": pt_bins,
#        "targets": [wp80_target]*6,
#        "match_boundary": True
#        }
#
#cfg["working_points"]["Fall17IsoV2"]["mvaEleID-Fall17-iso-V2-wpLoose"] = {
#        "type": "constant_cut_sig_eff_targets",
#        "categories": ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
#        "targets": [0.9475128258132411, 0.9490611850552356, 0.9044423070825363, 0.9948310845914105, 0.9914069066168602, 0.9839520294782502], # From Spring16HZZ
#        "match_boundary": True
#        }

cfg["working_points"]["Summer_16_ID_ISO"]["mvaEleID-Summer16-ID-ISO-HZZ"] = {
    "type":           "constant_cut_sig_eff_targets",
    "categories":     ["EB1_5", "EB2_5", "EE_5", "EB1_10", "EB2_10", "EE_10"],
    "targets":        [0.8164452295491703, 0.803096754509744, 0.7437667128195914, 0.9744637118403502, 0.9668337528255658, 0.9662211869474129], # Spring16HZZ, combinedIso < 0.35
    "match_boundary": False
    }


#####################
# CMSSW configuration
#####################

cfg["cmssw_cff"] = {}
cfg["cmssw_cff"]["Summer_16_ID_ISO"] = {}

cfg["cmssw_cff"]["Summer_16_ID_ISO"] = {
        "producer_config_name": "mvaEleID_Summer16_ID_ISO_producer_config",
        "file_name": "mvaElectronID_Summer16_ID_ISO_cff.py",
        "mvaTag": "Summer16IdIso",
        "mvaClassName": "ElectronMVAEstimatorRun2",
        }
