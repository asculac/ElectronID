import sys
import os
from config import cfg
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from os.path import join
import uproot


cuts = {
    "EB1": "abs(scl_eta) < 0.800",
    "EB2": "abs(scl_eta) >= 0.800 & abs(scl_eta) < 1.479",
    "EE" : "abs(scl_eta) >= 1.479",
    "5"  : "ele_pt < 10.",
    "10" : "ele_pt >= 10",
    }


def get_category(location, ptrange):

    training_bin = -1

    if   location == 'EB1' and ptrange == '5' : training_bin = 0
    elif location == 'EB2' and ptrange == '5' : training_bin = 1
    elif location == 'EE'  and ptrange == '5' : training_bin = 2
    elif location == 'EB1' and ptrange == '10': training_bin = 3
    elif location == 'EB2' and ptrange == '10': training_bin = 4
    elif location == 'EE'  and ptrange == '10': training_bin = 5

    return training_bin


#================#
# Read root file #
#================#

def load_df(rootfile, branches, entrystop=None):

    ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
    ntuple_file = join(ntuple_dir, 'test.root')
    tree = uproot.open(ntuple_file)["ntuplizer/tree"]

    df = tree.pandas.df(set(branches + ["ele_pt", "scl_eta", "matchedToGenEle", "genNpu"]), entrystop=None)

    df = df.query(cfg["selection_base"])
    df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)

    df = df.query("y > -1")

    return df


plot_dir   = join("plots", "sig_bckg")
ntuple_dir = cfg['ntuple_dir'] + '/' + cfg['submit_version']
rootfile = ntuple_dir + '/test.root'
marko_root_18='/grid_mnt/data__data.polcms/cms/mkovac/MVA_trees/Data/EGamma/DY_2018_25_3_2019/test.root'
####
# signal background distribution
###

wps = ["Autumn18IDISOHZZ"]
branches = wps + ["Autumn18IdIsoBoVals", "Autumn18IdIsoBoRawVals", "matchedToGenEle"]

for idname in cfg["trainings"]:
    for training_bin in cfg["trainings"][idname]:

        df = load_df(rootfile, branches, entrystop=None)
        df = df.query(cfg["trainings"][idname][training_bin]["cut"])  #cut for bins pT and eta            
        sig = df.query("y == 1")
        bkg = df.query("y == 0")

        #Marko
        tree_marko = uproot.open(marko_root_18)["ntuplizer/tree"]
        print(tree_marko)

        df_marko = tree_marko.pandas.df(set( ["Autumn18IdIsoBoVals", "Autumn18IdIsoRawVals", "matchedToGenEle"]+ ["ele_pt", "scl_eta", "matchedToGenEle", "genNpu"]), entrystop=None)
        df_marko = df_marko.query(cfg["selection_base"])
        df_marko.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)
        df_marko = df_marko.query("y > -1")

        df_marko = df_marko.query(cfg["trainings"][idname][training_bin]["cut"])  #cut for bins pT and eta   
        sig_marko = df_marko.query("y == 1")
        bkg_marko = df_marko.query("y == 0")


        fig = plt.figure(figsize=(10.4, 8.8))
        plt.title("2018 Ana")
        #plt.yscale("log")
        plt.hist([sig["Autumn18IdIsoBoRawVals"], bkg["Autumn18IdIsoBoRawVals"]], bins=100, label=['signal', 'background'], alpha=0.7)
        plt.legend(loc="upper right", ncol=1)
        plt.savefig(join(plot_dir, "sig_bckg_Ana_{0}.png".format(training_bin)), bbox_inches='tight')
        plt.close()
        
        fig = plt.figure(figsize=(10.4, 8.8))
        plt.title("2018 Marko")
        #plt.yscale("log")
        plt.hist([sig_marko["Autumn18IdIsoRawVals"], bkg_marko["Autumn18IdIsoRawVals"]], bins=100, label=['signal', 'background'], alpha=0.7)
        plt.legend(loc="upper right", ncol=1)
        plt.savefig(join(plot_dir, "sig_bckg_Marko_{0}.png".format(training_bin)), bbox_inches='tight')
        plt.close()
        
        fig = plt.figure(figsize=(10.4, 8.8))
        plt.title("2018 both")
        #plt.yscale("log")
        plt.xlim([-10, 10])
        if training_bin == "EB1_10":
            plt.xlim([-0.5,0.5])
        plt.hist([sig_marko["Autumn18IdIsoRawVals"], bkg_marko["Autumn18IdIsoRawVals"], sig["Autumn18IdIsoBoRawVals"], bkg["Autumn18IdIsoBoRawVals"]],
         bins=200, label=['signal Marko','background Marko','signal Ana', 'background Ana'], alpha=0.7, histtype=u'step')
        plt.legend(loc="upper left", ncol=1)
        plt.savefig(join(plot_dir, "sig_bckg_both_{0}.png".format(training_bin)), bbox_inches='tight')
        plt.close() 