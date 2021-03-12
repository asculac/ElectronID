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



plot_dir   = join("plots", "bkg")
rootfile_16 = '/grid_mnt/vol_home/llr/cms/skulac/RUN_2/Data/EGamma/DY_2016_5_12_2018/train_ana.root'
marko_root_16 = '/grid_mnt/vol_home/llr/cms/skulac/RUN_2/Data/EGamma/DY_2016_5_12_2018/train.root'

rootfile_18 = '/grid_mnt/data__data.polcms/cms/asculac/MVA_trees/Data/EGamma/DY_2018_22_02_2021/train.root'
marko_root_18='/grid_mnt/data__data.polcms/cms/mkovac/MVA_trees/Data/EGamma/DY_2018_25_3_2019/train.root'

tree = uproot.open(rootfile_18)["ntuplizer/tree"]

marko_tree = uproot.open(marko_root_18)["ntuplizer/tree"]

df = tree.pandas.df(set(["nEvent", "ele_pt", "scl_eta", "matchedToGenEle", "genNpu"])
            # "ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi",
            # "ele_oldcircularity", "ele_oldr9", "ele_scletawidth",
            # "ele_sclphiwidth", "ele_oldhe", "ele_kfhits", "ele_kfchi2",
            # "ele_gsfchi2", "ele_fbrem", "ele_gsfhits",
            # "ele_expected_inner_hits", "ele_conversionVertexFitProbability",
            # "ele_ep", "ele_eelepout", "ele_IoEmIop", "ele_deltaetain",
            # "ele_deltaphiin", "ele_deltaetaseed", "rho", "ele_psEoverEraw",
            # "ele_pfPhotonIso", "ele_pfChargedHadIso", "ele_pfNeutralHadIso"])
            ,entrystop=None)
df_marko = marko_tree.pandas.df(set(["nEvent", "ele_pt", "scl_eta", "matchedToGenEle", "genNpu"])
            # "ele_oldsigmaietaieta", "ele_oldsigmaiphiiphi",
            # "ele_oldcircularity", "ele_oldr9", "ele_scletawidth",
            # "ele_sclphiwidth", "ele_oldhe", "ele_kfhits", "ele_kfchi2",
            # "ele_gsfchi2", "ele_fbrem", "ele_gsfhits",
            # "ele_expected_inner_hits", "ele_conversionVertexFitProbability",
            # "ele_ep", "ele_eelepout", "ele_IoEmIop", "ele_deltaetain",
            # "ele_deltaphiin", "ele_deltaetaseed", "rho", "ele_psEoverEraw", 
            # "ele_pfPhotonIso", "ele_pfChargedHadIso", "ele_pfNeutralHadIso"])
            ,entrystop=None)

df = df.query(cfg["selection_base"])
df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)
df_marko = df.query(cfg["selection_base"])
df_marko.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)

sig = df.query("y == 1")
bkg = df.query("y == 0")
sig_marko = df_marko.query("y == 1")
bkg_marko = df_marko.query("y == 0")

print df[df['nEvent']==1000]['ele_pt']
print df["ele_pt"].loc[(df["nEvent"] == 1000)]
print df_marko["ele_pt"].loc[(df_marko["nEvent"] == 1000)]
print df["ele_pt"].loc[(df["nEvent"]==1000)]==df_marko["ele_pt"].loc[(df_marko["nEvent"] == 1000)]
'''
fig = plt.figure(figsize=(10.4, 8.8))
#ax.ylim(200)
#ax =[bkg["ele_pt"].plot()
#bkg_marko["ele_pt"].plot(ax=ax)
plt.xlim([0, 200])
#plt.ylim([0, 1000])
plt.hist([bkg["ele_pt"],bkg_marko["ele_pt"]],bins=50, histtype=u'step', label=['Ana bkg','Marko bkg'], alpha=1)  #bins=np.linspace(0,1000,50)
plt.legend(loc="upper right", ncol=1)
plt.savefig(join(plot_dir, "compare_pt_bkg.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([-3, 3])
plt.hist(bkg["scl_eta"], bins=50, histtype='step', label=['Ana bkg'], alpha=1)
plt.hist(bkg_marko["scl_eta"], bins=50, histtype='step', label=['Marko bkg'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_eta.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([-3, 3])
plt.hist([bkg["scl_eta"],bkg_marko["scl_eta"]],bins=50, histtype=u'step', label=['Ana bkg','Marko bkg'], alpha=1)  #bins=np.linspace(0,1000,50)
plt.legend(loc="upper right", ncol=1)
plt.savefig(join(plot_dir, "compare_eta_2.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,0.06])
#plt.hist([sig["ele_oldsigmaietaieta"], [bkg["ele_oldsigmaietaieta"],sig_marko["ele_oldsigmaietaieta"], bkg_marko["ele_oldsigmaietaieta"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_oldsigmaietaieta"], bkg_marko["ele_oldsigmaietaieta"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_oldsigmaietaieta.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
#plt.xlim([-3,3])
#plt.hist([sig["ele_oldsigmaiphiiphi"], [bkg["ele_oldsigmaiphiiphi"],sig_marko["ele_oldsigmaiphiiphi"], bkg_marko["ele_oldsigmaiphiiphi"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_oldsigmaiphiiphi"], bkg_marko["ele_oldsigmaiphiiphi"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_oldsigmaiphiiphi.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([-3,3])
#plt.hist([sig["ele_oldcircularity"], [bkg["ele_oldcircularity"],sig_marko["ele_oldcircularity"], bkg_marko["ele_oldcircularity"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_oldcircularity"], bkg_marko["ele_oldcircularity"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_oldcircularity.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,1.5])
#plt.hist([sig["ele_oldr9"], [bkg["ele_oldr9"],sig_marko["ele_oldr9"], bkg_marko["ele_oldr9"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_oldr9"], bkg_marko["ele_oldr9"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_oldr9.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,0.5])
#plt.hist([sig["ele_scletawidth"], [bkg["ele_scletawidth"],sig_marko["ele_scletawidth"], bkg_marko["ele_scletawidth"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_scletawidth"], bkg_marko["ele_scletawidth"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_scletawidth.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,0.5])
#plt.hist([sig["ele_sclphiwidth"], [bkg["ele_sclphiwidth"],sig_marko["ele_sclphiwidth"], bkg_marko["ele_sclphiwidth"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_sclphiwidth"], bkg_marko["ele_sclphiwidth"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_sclphiwidth.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,100])
#plt.hist([sig["ele_oldhe"], [bkg["ele_oldhe"],sig_marko["ele_oldhe"], bkg_marko["ele_oldhe"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_oldhe"], bkg_marko["ele_oldhe"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_oldhe.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
#plt.xlim([-3,3])
#plt.hist([sig["ele_kfhits"], [bkg["ele_kfhits"],sig_marko["ele_kfhits"], bkg_marko["ele_kfhits"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_kfhits"], bkg_marko["ele_kfhits"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_kfhits.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,6])
#plt.hist([sig["ele_expected_inner_hits"], [bkg["ele_expected_inner_hits"],sig_marko["ele_expected_inner_hits"], bkg_marko["ele_expected_inner_hits"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_expected_inner_hits"], bkg_marko["ele_expected_inner_hits"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_expected_inner_hits.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
#plt.xlim([-3,3])
#plt.hist([sig["ele_kfchi2"], [bkg["ele_kfchi2"],sig_marko["ele_kfchi2"], bkg_marko["ele_kfchi2"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_kfchi2"], bkg_marko["ele_kfchi2"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_kfchi2.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,50])
#plt.hist([sig["ele_gsfchi2"], [bkg["ele_gsfchi2"],sig_marko["ele_gsfchi2"], bkg_marko["ele_gsfchi2"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_gsfchi2"], bkg_marko["ele_gsfchi2"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_gsfchi2.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([-3,3])
#plt.hist([sig["ele_fbrem"], [bkg["ele_fbrem"],sig_marko["ele_fbrem"], bkg_marko["ele_fbrem"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_fbrem"], bkg_marko["ele_fbrem"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_fbrem.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([5,20])
#plt.hist([sig["ele_gsfhits"], [bkg["ele_gsfhits"],sig_marko["ele_gsfhits"], bkg_marko["ele_gsfhits"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_gsfhits"], bkg_marko["ele_gsfhits"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "ele_gsfhits.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([-1.5,-0.75])
#plt.hist([sig["ele_conversionVertexFitProbability"], [bkg["ele_conversionVertexFitProbability"],sig_marko["ele_conversionVertexFitProbability"], bkg_marko["ele_conversionVertexFitProbability"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_conversionVertexFitProbability"], bkg_marko["ele_conversionVertexFitProbability"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_conversionVertexFitProbability.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,7.5])
#plt.hist([sig["ele_ep"], [bkg["ele_ep"],sig_marko["ele_ep"], bkg_marko["ele_ep"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_ep"], bkg_marko["ele_ep"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_ep.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
#plt.xlim([-3,3])
#plt.hist([sig["ele_eelepout"], [bkg["ele_eelepout"],sig_marko["ele_eelepout"], bkg_marko["ele_eelepout"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_eelepout"], bkg_marko["ele_eelepout"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_eelepout.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([-1,1])
#plt.hist([sig["ele_IoEmIop"], [bkg["ele_IoEmIop"],sig_marko["ele_IoEmIop"], bkg_marko["ele_IoEmIop"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_IoEmIop"], bkg_marko["ele_IoEmIop"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_IoEmIop.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
#plt.xlim([-3,3])
#plt.hist([sig["ele_deltaetain"], [bkg["ele_deltaetain"],sig_marko["ele_deltaetain"], bkg_marko["ele_deltaetain"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_deltaetain"], bkg_marko["ele_deltaetain"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_deltaetain.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,0.25])
#plt.hist([sig["ele_deltaphiin"], [bkg["ele_deltaphiin"],sig_marko["ele_deltaphiin"], bkg_marko["ele_deltaphiin"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_deltaphiin"], bkg_marko["ele_deltaphiin"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_deltaphiin.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,0.100])
#plt.hist([sig["ele_deltaetaseed"], [bkg["ele_deltaetaseed"],sig_marko["ele_deltaetaseed"], bkg_marko["ele_deltaetaseed"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_deltaetaseed"], bkg_marko["ele_deltaetaseed"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_deltaetaseed.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,60])
#plt.hist([sig["rho"], [bkg["rho"],sig_marko["rho"], bkg_marko["rho"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["rho"], bkg_marko["rho"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_rho.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,3])
#plt.hist([sig["ele_psEoverEraw"], [bkg["ele_psEoverEraw"],sig_marko["ele_psEoverEraw"], bkg_marko["ele_psEoverEraw"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_psEoverEraw"], bkg_marko["ele_psEoverEraw"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_psEoverEraw.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0, 50])
#plt.hist([sig["ele_pfPhotonIso"], [bkg["ele_pfPhotonIso"],sig_marko["ele_pfPhotonIso"], bkg_marko["ele_pfPhotonIso"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_pfPhotonIso"], bkg_marko["ele_pfPhotonIso"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_pfPhotonIso.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,20])
#plt.hist([sig["ele_pfChargedHadIso"], [bkg["ele_pfChargedHadIso"],sig_marko["ele_pfChargedHadIso"], bkg_marko["ele_pfChargedHadIso"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_pfChargedHadIso"], bkg_marko["ele_pfChargedHadIso"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_pfChargedHadIso.png"), bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10.4, 8.8))
plt.xlim([0,20])
#plt.hist([sig["ele_pfNeutralHadIso"], [bkg["ele_pfNeutralHadIso"],sig_marko["ele_pfNeutralHadIso"], bkg_marko["ele_pfNeutralHadIso"]],bins=50, histtype=u'step', label=['Ana sig','Ana bkg', 'Marko sig', 'Marko bkg'], alpha=1)
plt.hist([bkg["ele_pfNeutralHadIso"], bkg_marko["ele_pfNeutralHadIso"]], bins=50, histtype=u'step', label=['Ana', 'Marko'], alpha=1)
plt.legend(loc="upper left", ncol=1)
plt.savefig(join(plot_dir, "compare_ele_pfNeutralHadIso.png"), bbox_inches='tight')
plt.close()
'''