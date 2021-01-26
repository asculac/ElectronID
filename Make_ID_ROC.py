import sys
import os
from config import cfg
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from os.path import join
import uproot
import pandas as pd
from sklearn import linear_model

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


#===============
# Read root file 
#===============

def load_df(rootfile, branches, entrystop=None):

   ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
   ntuple_file = join(ntuple_dir, 'test.root')
   tree = uproot.open(ntuple_file)["ntuplizer/tree"]

   df = tree.pandas.df(set(branches + ["ele_pt", "scl_eta", "matchedToGenEle", "genNpu"]), entrystop=entrystop)
    
   eff_area_file="/home/llr/cms/kovac/CMS/RUN_2/EGamma/2016/CMSSW_test/CMSSW_10_3_1/src/RecoEgamma/ElectronIdentification/data/Spring15/"
   ea = pd.read_csv(eff_area_file + "effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt", comment="#", delim_whitespace=True, header=None, names=["eta_min", "eta_max", "ea"])
	
   for i in range(len(ea))[::-1]:
      df.at[abs(df["scl_eta"]) < ea.iloc[i]["eta_max"], "ea"] = ea.iloc[i]["ea"]
      
   df["hzz_iso"] = -(df["ele_pfChargedHadIso"] + np.clip(df["ele_pfNeutralHadIso"] + df["ele_pfPhotonIso"] - df["rho"]*df["ea"], 0, None)) / df["ele_pt"]


   df = df.query(cfg["selection_base"])
   df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)
   df = df.query("y > -1")
      
   return df


#============
# Set folders
#============

plot_dir   = join("plots", cfg['submit_version'])
ntuple_dir = cfg['ntuple_dir'] + '/' + cfg['submit_version']
rootfile   = ntuple_dir + '/test.root'

if not os.path.exists(join(plot_dir, "ROC_ISO_after_ID")):
   os.makedirs(join(plot_dir, "ROC_ISO_after_ID"))


#====================================
# Enable or disable performance plots
#====================================

ROC    = True

#nmax = 1000000
nmax = None


#=================
# Other parameters
#=================

ylim_sig = 0, 100
ylim_bkg = 0.3001, 50

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors + ['k'] * 20

for i in range(6):
   colors[6+i] = colors[i]

roccolors = prop_cycle.by_key()['color']
roccolors[2] = "#ff7429"
roccolors[3] = "#79af67"
roccolors[0] = 'k'
roccolors[1] = '#0f4da5'

refcolors = ['#17becf'] * 3 + ['#bcbd22'] * 3

wps = ["Summer16IDISOHZZ", "Spring16HZZV1wpLoose"]

plot_args = [
        {"linewidth": 1, "color" : colors[0] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': '2016 ID+ISO'  },
        {"linewidth": 1, "color" : colors[1] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': '2016 ID only'},
        {"linewidth": 1, "color" : colors[2] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': '2017 ID+ISO V2'},
        ]

plot_args_bkg = []

for i in range(len(plot_args)):
    plot_args_bkg.append(plot_args[i].copy())
    plot_args_bkg[i]["label"] = ''

plot_args = [plot_args, plot_args_bkg]

roc_curves = [
              ("2016", "hzz_iso", "2016 ISO after ID"),
              ("2016", "Spring16HZZV1RawVals", "2016 ID only"),
              ("2016", "Summer16IdIsoRawVals", "2016 ID+ISO"),
             ]

roc_plot_args = {
             'markeredgewidth': 0,
             'linewidth': 2,
            }


#===============
# Plot functions
#===============

def create_axes(yunits=4):
    
   fig = plt.figure(figsize=(6.4, 4.8))
   gs = gridspec.GridSpec(yunits, 1)
   ax1 = plt.subplot(gs[:2, :])
   ax2 = plt.subplot(gs[2:, :])
   axarr = [ax1, ax2]

   gs.update(wspace=0.025, hspace=0.075)

   plt.setp(ax1.get_xticklabels(), visible=False)

   ax1.grid()
   ax2.grid()

   return ax1, ax2, axarr
   
    
def plot_roc(df, x, label = None):
   fpr, tpr, _ = metrics.roc_curve(df["y"] == 1, df[x], pos_label = 1)
   
   fpr = fpr*100
   tpr = tpr*100
   
   plt.semilogy(tpr[tpr > 70], fpr[tpr > 70], label = x if label is None else label)
    


#===========
# ROC Curves
#===========

if ROC:

   print("Making ROC curves")
   
   branches = ["Summer16IdIsoRawVals", "Spring16HZZV1RawVals", "Spring16HZZV1wpLoose", "ele_pfChargedHadIso", "ele_pfNeutralHadIso", "ele_pfPhotonIso", "rho"]
   df = load_df(rootfile, branches, entrystop=nmax)
    
   for ptrange in ["5"]:
#   for ptrange in ["5", "10"]:

      for location in ["EB1", "EB2"]:   
#      for location in ['EB1', 'EB2', 'EE']:

         print("[INFO] Processing {0} {1}...".format(location, ptrange))
            
         df_eta_pt = df.query(cfg["trainings"]["Spring_16_ID_ISO"]["_".join([location, ptrange])]["cut"])
         sig = df_eta_pt.query("y == 1")
         bkg = df_eta_pt.query("y == 0")
            
         n_sig = len(sig)
         n_bkg = len(bkg)
               
         grid_df = pd.DataFrame()

         q = np.linspace(0, 0.8, 50)[1:-1]
         iso_quantiles = df_eta_pt.hzz_iso.quantile(q)

         i = 0
         for cut in iso_quantiles:
            df_cut = df_eta_pt[df_eta_pt.hzz_iso > cut]
                        
            sig_eff = len(df_cut.query("y == 1")) * 1./n_sig
            bkg_eff = len(df_cut.query("y == 0")) * 1./n_bkg
            
#            print cut
#            print sig_eff
#            print bkg_eff
#            print df_cut["Spring16HZZV1RawVals"]
            
            fpr, tpr, _ = metrics.roc_curve(df_cut["y"] == 1, df_cut["Spring16HZZV1RawVals"], pos_label=1)
            tpr = sig_eff * tpr * 100
            fpr = bkg_eff * fpr * 100
            
#            if i % 2 == 0:
#               plt.semilogy(tpr[tpr > 70], fpr[tpr > 70], label="{0:.2f}".format(-cut))
            
            tmp_df = pd.DataFrame(data={"tpr":tpr, "fpr":fpr})
            grid_df = pd.concat([grid_df, tmp_df])
            i = i+1
         
         tpr_bins = np.linspace(70, 100, 200)
         grid_df["tpr_bin"] = pd.cut(grid_df["tpr"], tpr_bins)
         fpr = grid_df.groupby("tpr_bin").fpr.min().values
         tpr = (tpr_bins[1:]+tpr_bins[:-1])/2.
         
         plot_roc(df_eta_pt, "Spring16HZZV1RawVals", label = "2016 ID only")
         plot_roc(df_eta_pt, "Summer16IdIsoRawVals", label = "2016 ID with ISO")
         plt.plot(tpr, fpr, label = "2016 ISO after ID")
         plt.xlim(70, 100)
         plt.ylim(0.101, 100)
         plt.xlabel("Signal efficiency")
         plt.ylabel("Background efficiency")
         plt.legend(loc="upper left", ncol=2)
#         plt.show()
         plt.savefig(join(plot_dir, "ID_after_ISO_ROC/2016_{0}_{1}.pdf".format(location, ptrange)), bbox_inches='tight')
         plt.savefig(join(plot_dir, "ID_after_ISO_ROC/2016_{0}_{1}.eps".format(location, ptrange)), bbox_inches='tight')
         os.system("convert -density 150 -quality 100 " + join(plot_dir, "ID_after_ISO_ROC/2016_{0}_{1}.eps".format(location, ptrange)) + " "
                                                        + join(plot_dir, "ID_after_ISO_ROC/2016_{0}_{1}.png".format(location, ptrange)))
         
         plt.close()
            
            
#            ax1, ax2, axes = create_axes(yunits=3)
#
#            xmin = 60
#
#            yref, xref, _ = metrics.roc_curve(df["y"] == 1, df["Spring16HZZV1RawVals"])
#            xref = xref * 100
#            yref = yref * 100
#
#            k = 0
#
#            for yr, cl, lbl in roc_curves:        
#                
#                if cl == "hzz_iso":
#                   y, x, _ = metrics.roc_curve(df_id["y"] == True, df_id[cl])
#                   x = 1 - x
#                   y = 1 - y
#                
#                   x = x * eff_sig * 100
#                   y = y * eff_bkg * 100
#                
#                else:
#                   y, x, _ = metrics.roc_curve(df["y"] == True, df[cl])
#                   x = x * 100
#                   y = y * 100                 
#
##                y, x, _ = metrics.roc_curve(df["y"] == True, df[cl])
##                x = x * 100
##                y = y * 100 
#
#                sel = x > xmin
#
#                ax1.semilogy(x[sel], y[sel], color=roccolors[k+1], label=lbl, **roc_plot_args)
#                ax2.plot(x[sel], y[sel] / np.interp(x[sel], xref, yref), color=roccolors[k+1], **roc_plot_args)
#                
#                k = k + 1
#
#            # Styling the plot
#            ax1.set_ylabel(r'Background efficiency [%]')
#
#            ax2.set_xlabel(r'Signal efficiency [%]')
#            ax2.set_ylabel(r'Ratio')
#
#            ax1.set_ylim(0.101, 100)
#            ax2.set_ylim(0.201, 1.09)
#
#            ax1.legend(loc="upper left", ncol=2)
#
#            ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
#
#            plt.savefig(join(plot_dir, "ROC_ISO_after_ID/2016_{0}_{1}.pdf".format(location, ptrange)), bbox_inches='tight')
#            plt.savefig(join(plot_dir, "ROC_ISO_after_ID/2016_{0}_{1}.eps".format(location, ptrange)), bbox_inches='tight')
#            os.system("convert -density 150 -quality 100 " + join(plot_dir, "ROC_ISO_after_ID/2016_{0}_{1}.eps".format(location, ptrange)) + " "
#                                                           + join(plot_dir, "ROC_ISO_after_ID/2016_{0}_{1}.png".format(location, ptrange)))
#
#            plt.close()