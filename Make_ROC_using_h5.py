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
import pandas

plot_dir = join("plots", cfg['submit_version'])

def load_df(h5_file):
   
   df = pandas.read_hdf(h5_file)

#  df = df.query(cfg["selection_base"])
#  df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)

   df = df.query("y > -1")

   return df


################
# Basic settings
################

h5_dir = cfg['out_dir'] + '/' + cfg['submit_version']

if not os.path.exists(join(plot_dir, "ROC_h5")):
   os.makedirs(join(plot_dir, "ROC_h5"))

# Enable or disable performance plots
ROC = True

##################
# Other parameters
##################

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors + ['k'] * 20

for i in range(6):
   colors[6+i] = colors[i]

roccolors = prop_cycle.by_key()['color']
roccolors[2] = roccolors[0]
roccolors[3] = roccolors[1]
roccolors[0] = 'k'
roccolors[1] = '#7f7f7f'

refcolors = ['#17becf'] * 3 + ['#bcbd22'] * 3

plot_args = [
        {"linewidth": 1, "color" : colors[0] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp90'          },
        {"linewidth": 1, "color" : colors[1] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp80'          },
        {"linewidth": 1, "color" : colors[2] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wpLoose'       },
        {"linewidth": 1, "color" : colors[3] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp90 w/ iso'   },
        {"linewidth": 1, "color" : colors[4] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wp80 w/ iso'   },
        {"linewidth": 1, "color" : colors[5] , "markersize": 2, "marker": 'o', "linestyle": '-' , 'label': 'wpLoose w/ iso'},
        {"linewidth": 0.5, "color" : colors[6] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[7] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[8] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[9] , "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[10], "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        {"linewidth": 0.5, "color" : colors[11], "markersize": 1, "marker": '.', "linestyle": '--', 'label': ''              },
        ]

plot_args_bkg = []

for i in range(len(plot_args)):
   plot_args_bkg.append(plot_args[i].copy())
   plot_args_bkg[i]["label"] = ''

plot_args = [plot_args, plot_args_bkg]

roc_curves = [
              ("2016", "bdt_score_default", "ID+ISO XGBoost"),
              ("2016", "bdt_score_optimized", "ID+ISO XGBoost + BO")
             ]

roc_plot_args = {
                 'markeredgewidth': 0,
                 'linewidth': 2,
                }


##################
# Helper functions
##################

def create_axes(yunits=4):
   fig = plt.figure(figsize=(6.4, 4.8))
   gs = gridspec.GridSpec(yunits, 1)
   ax1 = plt.subplot(gs[:2, :])
   ax2 = plt.subplot(gs[2:, :])
   axarr = [ax1, ax2]

   gs.update(wspace=0.025, hspace=0.075)

   plt.setp(ax1.get_xticklabels(), visible = False)

   ax1.grid()
   ax2.grid()

   return ax1, ax2, axarr


############
# ROC Curves
############

if ROC:

   print("Making ROC curves")

   for ptrange in ["5", "10"]:
      
      for location in ["EB1", "EB2", "EE"]:

         print("processing {0} {1}...".format(location, ptrange))

         h5_file = h5_dir + "/Summer_16_ID_ISO/" + location + "_" + ptrange + '/pt_eta_score.h5'
            
         df = load_df(h5_file)

         ax1, ax2, axes = create_axes(yunits=3)

         xmin = 70

         yref, xref, _ = metrics.roc_curve(df["y"] == 1, df["bdt_score_default"])
         xref = xref * 100
         yref = yref * 100

         k = 0
         for yr, cl, lbl in roc_curves:

            y, x, _ = metrics.roc_curve(df["y"] == 1, df[cl])
            x = x * 100
            y = y * 100

            sel = x > xmin

            ax1.semilogy(x[sel], y[sel], color=roccolors[k], label=lbl, **roc_plot_args)
            ax2.plot(x[sel], y[sel] / np.interp(x[sel], xref, yref), color=roccolors[k], **roc_plot_args)

            k = k + 1

         # Styling the plot
         ax1.set_ylabel(r'Background efficiency [%]')

         ax2.set_xlabel(r'Signal efficiency [%]')
         ax2.set_ylabel(r'Ratio')

         ax1.set_ylim(0.101, 100)
         ax2.set_ylim(0.301, 1.09)

         ax1.legend(loc="upper left", ncol=1)

         ax1.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

         plt.savefig(join(plot_dir, "ROC_h5/2016_{0}_{1}.pdf".format(location, ptrange)), bbox_inches='tight')
         plt.savefig(join(plot_dir, "ROC_h5/2016_{0}_{1}.eps".format(location, ptrange)), bbox_inches='tight')
         os.system("convert -density 150 -quality 100 " + join(plot_dir, "ROC_h5/2016_{0}_{1}.eps".format(location, ptrange)) + " "
                                                        + join(plot_dir, "ROC_h5/2016_{0}_{1}.png".format(location, ptrange)))

         plt.close()