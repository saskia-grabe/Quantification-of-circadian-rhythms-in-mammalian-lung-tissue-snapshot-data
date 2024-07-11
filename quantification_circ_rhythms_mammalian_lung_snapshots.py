#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:22:08 2022

@author: saskia
"""
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import scipy as sc

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from sklearn import decomposition

import os
os.chdir('your/directory')
#%% MidpointNormalize


class MidpointNormalize(mpl.colors.Normalize):
    '''
        icemtel's function to set a middle point of a colormap
        https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    '''
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sc.ma.masked_array(np.interp(value, x, y))
    
#%% funcs for fitting

def oscifunc(t, a, b, c):
    return a*np.sin((2.*np.pi/24.)*t + (2.*np.pi/24.)*b) + c


def oscifunc3(t, c, a, b, d, e):
    # based on fourier series
    return c + a*np.sin((2.*np.pi/24.)*t) + b*np.cos((2.*np.pi/24.)*t) + d*np.sin((4.*np.pi/24.)*t) + e*np.cos((4.*np.pi/24.)*t)

#%% general math funcs

def r_squared(time, y, func, popt):
    rss        = np.sum((y - func(time, *popt))**2.)
    ss_tot     = np.sum((y - np.mean(y))**2.)
    r_squared  = 1. - (rss/ss_tot)
    return r_squared


#%% funcs for rank-based gene lists
    
def nonrandomrs(querygenes, allgenes, data, time, func=oscifunc3, p0=np.array([1., 1., 1., 1., 1.]), startgene='PER3'):
    time2 = np.arange(min(time), max(time)+12., 0.1)
    day = time2[-1] - 24.
    y0 = data.iloc[np.where(allgenes == startgene)[0][0]].values
    popt, pcov = curve_fit(func, time[~(np.isnan(y0)|np.isnan(time))], y0[~(np.isnan(y0)|np.isnan(time))], p0)
    fy0 = func(time2, *popt)
    
    spearcorrs  = []
    phases      = []
    geneindices = []
    r2      = []
    amps    = []
    for i, w in enumerate(querygenes):
        if len(np.where(allgenes == w)[0]) > 0:
            ind = np.where(allgenes == w)[0][0]
            v   = data.iloc[ind].values   
            spearcorr = sc.stats.spearmanr(y0[~(np.isnan(y0)|np.isnan(v))], v[~(np.isnan(y0)|np.isnan(v))])[0]
            try:
                popt, pcov = curve_fit(func, time[~(np.isnan(v)|np.isnan(time))], v[~(np.isnan(v)|np.isnan(time))], p0)
                y = func(time2, *popt)
                phase = time2[time2>day][fy0[time2>day]==max(fy0[time2>day])] - time2[time2>day][y[time2>day]==max(y[time2>day])]
                if phase[0] < 0.: phase[0] += 24.
                phases.append(phase[0])
                spearcorrs.append(spearcorr)
                geneindices.append(i)
                r2i = r_squared(time, v, func, popt)
                r2.append(r2i)
                
                amps.append((max(y) - min(y))/2.)
                
            except:
                pass
    spearcorrs  = np.asanyarray(spearcorrs)
    phases      = np.asanyarray(phases)
    geneindices = np.asanyarray(geneindices)
    r2          = np.asanyarray(r2)
    amps        = np.asanyarray(amps)
    return spearcorrs, phases, geneindices, r2, amps


def in_and_out_of_phase(querygenename, allgenes, querygenes, data):
    gene = data.iloc[np.where(allgenes == querygenename)[0][0]].values       
    
    spearcorrs = []
    genenames  = []
    for i, w in enumerate(querygenes):
        if len(np.where(allgenes == w)[0]) > 0:
            ind = np.where(allgenes == w)[0][0]
            v   = data.iloc[ind].values   
            spearcorr = sc.stats.spearmanr(gene, v)[0]
            spearcorrs.append(spearcorr)
            genenames.append(w)
    spearcorrs = np.asanyarray(spearcorrs)
    genenames  = np.asanyarray(genenames) 
    return spearcorrs, genenames

def genelist_to_csv(genes0h, genes12h, genes6h, filename):
    
    for i, v in enumerate([genes0h, genes12h, genes6h]):
        if len(v) != max(len(genes0h), len(genes12h), len(genes6h)):
            v.extend(['']*(max(len(genes0h), len(genes12h), len(genes6h))-len(v)))

    pd.DataFrame({'0h':genes0h, '12h':genes12h, '6h':genes6h}).to_csv ('results/'+filename+'.csv', index=None, header=True)

#%% TCGA specific funcs

def correlationmatrix_tcga(querygenes, allgenes, data, normalfiles, tumorfiles):
    arrnegposneu = np.zeros((len(querygenes), 565))
    for i, v in enumerate(querygenes):
        if len(np.where(allgenes == v)[0]) > 0:
            ind = np.where(allgenes == v)[0][0]
            arrnegposneu[i] = data.iloc[ind].values
        else:
            arrnegposneu[i] *= np.nan

    normalarr = []
    for i, v in enumerate(normalfiles):
        if np.any(arrnegposneu.T[v] > 10.):
            pass
        else:
            normalarr.append(arrnegposneu.T[v])
    normalarr = np.asanyarray(normalarr).T
    normrs = np.zeros((len(normalarr), len(normalarr)))
    for i in range(len(querygenes)):
        for j in range(len(querygenes)):
            normrs[i][j] = sc.stats.spearmanr(normalarr[i], normalarr[j])[0]
            
    tumorarr = []
    for i, v in enumerate(tumorfiles):
        if np.any(arrnegposneu.T[v] > 10.):
            pass
        else:
            tumorarr.append(arrnegposneu.T[v])
    tumorarr = np.asanyarray(tumorarr).T
    tumorrs = np.zeros((len(tumorarr), len(tumorarr)))
    for i in range(len(querygenes)):
        for j in range(len(querygenes)):
            tumorrs[i][j] = sc.stats.spearmanr(tumorarr[i], tumorarr[j])[0]
            
    return normrs, tumorrs


def correlationmatrix_tcga_paired(querygenes, allgenes, data, normalfiles, tumorfiles, normalmatchedfiles, tumormatchedfiles):
    
    arrnegposneu = np.zeros((len(querygenes), 565))
    for i, v in enumerate(querygenes):
        if len(np.where(tcgaluadgenes == v)[0]) > 0:
            ind = np.where(tcgaluadgenes == v)[0][0]
            arrnegposneu[i] = tcgaluad.iloc[ind].values
        else:
            arrnegposneu[i] *= np.nan
            
    normalarr = []
    for i, v in enumerate(normalmatchedfiles):
        if np.any(arrnegposneu.T[v] > 10.):
            pass
        else:
            normalarr.append(arrnegposneu.T[v])
    normalarr = np.asanyarray(normalarr).T
    normrs = np.zeros((len(querygenes), len(querygenes)))
    for i in range(len(querygenes)):
        for j in range(len(querygenes)):
            normrs[i][j] = sc.stats.spearmanr(normalarr[i], normalarr[j])[0]
    
            
    tumorarr = []
    for i, v in enumerate(tumormatchedfiles):
        if np.any(arrnegposneu.T[v] > 10.):
            pass
        else:
            tumorarr.append(arrnegposneu.T[v])
    tumorarr = np.asanyarray(tumorarr).T
    tumorrs = np.zeros((len(querygenes), len(querygenes)))
    for i in range(len(querygenes)):
        for j in range(len(querygenes)):
            tumorrs[i][j] = sc.stats.spearmanr(tumorarr[i], tumorarr[j])[0]
            
    return normrs, tumorrs


#%% correlationmatrix funcs

def correlationmatrix(querygenes, allgenes, data):
    rs = []
    for i, v in enumerate(querygenes):
        try:
            spearcorrs = []
            query = data.iloc[np.where(allgenes == v)[0][0]].values
            for j, w in enumerate(querygenes):
                try:
                    current = data.iloc[np.where(allgenes == w)[0][0]].values
                    spearcorr = sc.stats.spearmanr(query, current, nan_policy='omit')[0]
                    spearcorrs.append(spearcorr)
                except:
                    spearcorrs.append(np.nan)
            rs.append(spearcorrs)
        except:
            rs.append(np.nan*np.zeros(len(querygenes)))
    rs = np.asanyarray(rs)
    
    return rs


def querygenearray(querygenes, allgenes, data):
    arr = np.zeros((len(querygenes), len(data.columns)))
    for i, v in enumerate(querygenes):
        if len(np.where(allgenes == v)[0]) > 0:
            ind = np.where(allgenes == v)[0][0]
            arr[i] = data.iloc[ind].values
        else:
            arr[i] = np.nan
    return arr


def pcaphases(pcacomponents):
    #1st and 2nd component
    center         = np.array([np.mean(pcacomponents.T[0]), np.mean(pcacomponents.T[1])])
    pcacomponents -= center.T
    pcaphases      = np.arctan2(pcacomponents.T[1], pcacomponents.T[0])
    
    return pcaphases

#%% load Zhang2014

data = pd.read_csv('circadb_mouse_lung/GSE54652-GPL6246_series_matrix.txt', sep='\t', skiprows=60, skipfooter=1)

#Intensity values normalized using Affymetrix Expression Console software (GC-RMA)
affynames = pd.read_csv('circadb_mouse_lung/mart_export_affymouse.txt', sep='\t')
affynames = affynames[['AFFY MoGene 1 0 st v1 probe', 'Gene name']]
affynamesdic = dict(zip(affynames['AFFY MoGene 1 0 st v1 probe'].tolist(), affynames['Gene name'].tolist()))

affycol = []
for i, v in enumerate(data['ID_REF'].tolist()):
    if v in affynames['AFFY MoGene 1 0 st v1 probe'].tolist():
        affycol.append(str(affynamesdic[v]).upper())
    else:
        affycol.append('UNKNOWN')
data['ID_REF'] = affycol


columns = ['ID_REF']
for i in range(206, 230):
    columns.append('GSM1321'+str(i))  # selects lung only
data = data[columns]

# get an array with all available genes
genes = np.asanyarray(data.reset_index()['ID_REF'])
data = data.set_index('ID_REF')
data = np.log2(data)

mousetime  = np.arange(18, 66, 2)

#%% load Esser2023
#young = 6 months, aged = 18 months, old = 27 months until release in DD
#Zhang's mice are 7 weeks old until release in DD

mouse_esser = pd.read_csv('esser2022/GSE201207_cpm_216.csv')

essergenes = np.char.upper(np.char.asarray(mouse_esser.copy().reset_index()['Unnamed: 0'])).astype('str')


esserlungyoung   = mouse_esser.loc[:, 'ct18-young-lung':'ct62-young-lung']
esserlungaged    = mouse_esser.loc[:, 'ct18-aged-lung':'ct62-aged-lung']
esserlungold     = mouse_esser.loc[:, 'ct18-old-lung':'ct62-old-lung']

esserlungyoung.index = essergenes.tolist()
esserlungaged.index  = essergenes.tolist()
esserlungold.index   = essergenes.tolist()

esserlungyoung = esserlungyoung[(esserlungyoung > 0).all(axis=1)]
esserlungaged  = esserlungaged[(esserlungaged > 0).all(axis=1)]
esserlungold   = esserlungold[(esserlungold > 0).all(axis=1)]
        
esserlungyoung = np.log2(esserlungyoung)
esserlungaged  = np.log2(esserlungaged)
esserlungold   = np.log2(esserlungold)

esserlungyounggenes = esserlungyoung.index.values
esserlungagedgenes  = esserlungaged.index.values
esserlungoldgenes   = esserlungold.index.values

mousetime2 = np.arange(18, 66, 4)

#%% load TCGA-LUAD

humannames = pd.read_csv('TCGALUAD_data/human_gene_names_ensembl.txt', sep='\t')
humannames = humannames[['Gene stable ID', 'Gene name']]
humannamesdic = dict(zip(humannames['Gene stable ID'].tolist(), humannames['Gene name'].tolist()))


tcgaluad = pd.read_csv('TCGALUAD_data/alldata/all_tcgaluad.csv', index_col='gene')
tcgaluad = tcgaluad.drop(tcgaluad.columns[[0]], axis=1)

tcgaluad = tcgaluad[(tcgaluad > 0).all(axis=1)] #remove rows/genes when log2 is not applicable
tcgaluad_unprocessed = tcgaluad.copy()

#get an array with all available genes
tcgaluadgenes = np.asanyarray(tcgaluad.copy().reset_index()['gene'])
#remove ensemble gene version number
for i, v in enumerate(tcgaluadgenes): tcgaluadgenes[i] = v[:15]

i = 0
for column in tcgaluad:
    if i != 0:
        tcgaluad[column] = np.log2(tcgaluad[column])
        m = tcgaluad[column].median()
        tcgaluad[column] = tcgaluad[column] - m
    i += 1


tcgaluadgenes_ens = []
for i, v in enumerate(tcgaluadgenes):
    if v in humannames['Gene stable ID'].tolist():
        tcgaluadgenes_ens.append(str(humannamesdic[v]).upper())
    else:
        tcgaluadgenes_ens.append('UNKNOWN')
tcgaluad['gene'] = tcgaluadgenes_ens
tcgaluad         = tcgaluad.set_index('gene')
tcgaluadgenes    = np.asanyarray(tcgaluadgenes_ens)




samples = np.asanyarray(list(tcgaluad))

fnames = pd.read_csv('TCGALUAD_data/gdc_sample_sheet.2022-03-25.tsv', sep='	')

files  = fnames['File Name'].values
for i, v in enumerate(files):
    files[i] = v[:-16]

tumor  = np.asanyarray(fnames['Sample Type'].values)
caseid = np.asanyarray(fnames['Case ID'].values)
normalcaseid = np.asanyarray(fnames.loc[fnames['Sample Type'] == 'Solid Tissue Normal', 'Case ID'])
tumorcaseid  = np.asanyarray(fnames.loc[fnames['Sample Type'] == 'Primary Tumor', 'Case ID'])

normalfiles = []
tumorfiles  = []
tumormatchedfiles  = []
normalmatchedfiles = []
tumormatchedcaseids  = []
normalmatchedcaseids = []

caseidinsamples = []

for i, v in enumerate(samples): #samples has the same order as arr, but not as files
    if v in files: #why are there additional filenames -->check TCGA
        ind = np.argwhere(files==v)[0][0]
        if (tumor[ind] == 'Primary Tumor'):
            tumorfiles.append(i)
            if caseid[ind] in normalcaseid:
                tumormatchedfiles.append(i)
                tumormatchedcaseids.append(caseid[ind])
        elif tumor[ind] == 'Solid Tissue Normal':
            normalfiles.append(i)
            if caseid[ind] in tumorcaseid:
                normalmatchedfiles.append(i)
                normalmatchedcaseids.append(caseid[ind])
        caseidinsamples.append(caseid[ind])
        
#remove files, which are not in samples, indicated falsely as matched
tumormatchedfiles  = np.asanyarray([tumormatchedfiles])[0]
normalmatchedfiles = np.asanyarray([normalmatchedfiles])[0]
tumormatchedcaseids  = np.asanyarray([tumormatchedcaseids])[0]
normalmatchedcaseids = np.asanyarray([normalmatchedcaseids])[0]
unmatchedfiles = []

for i, v in enumerate(normalmatchedcaseids):
    if v not in tumormatchedcaseids and v  not in unmatchedfiles:
        for j, w in enumerate(np.where(normalmatchedcaseids==v)[0]):
            normalmatchedfiles = np.delete(normalmatchedfiles, np.where(normalmatchedcaseids==v)[0][0])
            normalmatchedcaseids = np.delete(normalmatchedcaseids, np.where(normalmatchedcaseids==v)[0][0])
        unmatchedfiles.append(v)
for i, v in enumerate(tumormatchedcaseids):
    if v not in normalmatchedcaseids and v  not in unmatchedfiles:
        for j, w in enumerate(np.where(tumormatchedcaseids==v)[0]):
            tumormatchedfiles = np.delete(tumormatchedfiles, np.where(tumormatchedcaseids==v)[0][0])
            tumormatchedcaseids = np.delete(tumormatchedcaseids, np.where(tumormatchedcaseids==v)[0][0])
        unmatchedfiles.append(v)
        
print('normalsamples: ', len(normalfiles))
print('tumorsamples: ', len(tumorfiles))
print('matched samples: ', len(tumormatchedfiles))
print('matched samples: ', len(normalmatchedfiles))

#tumormatchedcaseids --> some patients have multiple tumor samples

#mapping of normal and tumor for pcaphases
tumormatchedcaseids_index = np.zeros(len(tumormatchedcaseids))
for i, v in enumerate(normalmatchedcaseids):
    tumormatchedcaseids_index[np.where(tumormatchedcaseids==v)] = i

#%%  load baboon, Mure          
baboon = pd.read_csv('panda2018_baboon/GSE98965_baboon_tissue_expression_FPKM.csv', sep=',')
columns = ['Symbol']
for i in range(0, 10, 2): columns.append('LUN.ZT0'+str(i))#selects lung only
for i in range(10, 24, 2): columns.append('LUN.ZT'+str(i))
baboon = baboon[columns]
baboon = baboon.set_index('Symbol')

baboon = baboon[(baboon > 0).all(axis=1)] #remove rows/genes when log2 is not applicable
for column in baboon:
        baboon[column] = np.log2(baboon[column])

baboongenes = np.asanyarray(baboon.index)

baboontime = np.arange(0., 24., 2.)

#%% remove genes, which are not shared across all muscle datasets

allgenes = np.load('TCGALUAD_data/alldata/sharedgenelist.npy')
allgenes = np.unique(allgenes)

#%% simulated rs vs phi


norm = MidpointNormalize(vmin=-1., vmax=1., midpoint=0.)


fi2, ax2 = plt.subplots(1, 1, figsize=(15, 10))

phis = np.arange(0., 24.5, 0.5)
time = np.arange(18, 66, 4)

for i, phi in enumerate(phis):
    time2 = np.arange(min(time), max(time)+12., 0.1) + np.random.uniform(-24., 24.)
    x1 = np.sin((2.*np.pi/24.)*time2)
    x2 = np.sin((2.*np.pi/24.)*time2 + (2.*np.pi/24.)*phi)
    spearcorr = sc.stats.spearmanr(x1, x2)[0]

    ax2.scatter(spearcorr, phi, c=spearcorr, s=150., edgecolor='gray', cmap='seismic', norm=norm, alpha=0.8)

ax2.set_xlabel(r'$r_{S}$', fontsize=22)
ax2.set_ylabel(r'$\phi$ (h)', fontsize=22)
ax2.set_xlim(-1.1, 1.1)
ax2.set_xticks([-1., 0., 1.])
ax2.set_xticklabels([-1., 0., 1.], fontsize=18.)
ax2.set_yticks([0., 6., 12., 18., 24.])
ax2.set_yticklabels([0, 6, 12, 18, 24], fontsize=18.)
ax2.grid()


#%% in_and_out_of_phase lung tissue --> dataset specific circadian gene lists

norm = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap = mpl.colormaps['seismic']


fi = plt.figure(figsize=(29.7/2.54, 18/2.54), constrained_layout=True)
gs = GridSpec(2, 3, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:2])
ax3 = fi.add_subplot(gs[:1, 2:])
ax7 = fi.add_subplot(gs[1:, :1])
ax8 = fi.add_subplot(gs[1:, 1:2])
ax9 = fi.add_subplot(gs[1:, 2:])


ax =  [ax1, ax2, ax3]
ax2 = [ax7, ax8, ax9]


timel2 = [mousetime, mousetime2, mousetime2, mousetime2, baboontime]
datasets  = [[genes, data], [esserlungyounggenes, esserlungyoung], [esserlungagedgenes, esserlungaged], [esserlungoldgenes, esserlungold], [baboongenes, baboon]]
marker = ['o', 's', 'p', 'd', 'X']

for m, startgene in enumerate(['PER3']):

    in_out_phase       = []
    in_out_phase_rs    = []
    in_out_phase_6h    = []
    in_out_phase_6h_rs = []
    for i, v in enumerate(datasets):
        inandoutofphase = in_and_out_of_phase(startgene, v[0], v[0], v[1])
        ingenes  = inandoutofphase[1][inandoutofphase[0]>0.6]
        outgenes = inandoutofphase[1][inandoutofphase[0]<-0.6]
        in_out_phase.append([ingenes, outgenes])
        ingenesrs  = inandoutofphase[0][inandoutofphase[0]>0.6]
        outgenesrs = inandoutofphase[0][inandoutofphase[0]<-0.6]
        in_out_phase_rs.append([ingenesrs, outgenesrs])
        
        #startgene2 is most antiphasic to statgene
        startgene2 = pd.unique(in_out_phase[i][1][np.argsort(in_out_phase_rs[i][1])])[:100][0]
        inandoutofphase2 = in_and_out_of_phase(startgene2, v[0], v[0], v[1])
        
        #intersecting 6h-genes of startgene and startgene2
        _, ind1, _ind2 = np.intersect1d(inandoutofphase[1][np.abs(inandoutofphase[0])<0.4], inandoutofphase2[1][np.abs(inandoutofphase2[0])<0.4], return_indices=True)
        
        rs_phi_gene_r2_amp = nonrandomrs(inandoutofphase[1][np.abs(inandoutofphase[0])<0.4][ind1], v[0], v[1], timel2[i], startgene=startgene)
        in_out_phase_6h.append(inandoutofphase[1][np.abs(inandoutofphase[0])<0.4][ind1][rs_phi_gene_r2_amp[3]>0.8][np.argsort(rs_phi_gene_r2_amp[3][rs_phi_gene_r2_amp[3]>0.8])[::-1]])
        in_out_phase_6h_rs.append(inandoutofphase[0][np.abs(inandoutofphase[0])<0.4][ind1][rs_phi_gene_r2_amp[3]>0.8][np.argsort(rs_phi_gene_r2_amp[3][rs_phi_gene_r2_amp[3]>0.8])[::-1]])
    
    mousecombi_in  = np.intersect1d(in_out_phase[0][0], in_out_phase[1][0])
    mousecombi_in  = np.intersect1d(mousecombi_in, in_out_phase[2][0])
    mousecombi_in  = np.intersect1d(mousecombi_in, in_out_phase[3][0])
    mousecombi_out = np.intersect1d(in_out_phase[0][1], in_out_phase[1][1])
    mousecombi_out = np.intersect1d(mousecombi_out, in_out_phase[2][1])
    mousecombi_out = np.intersect1d(mousecombi_out, in_out_phase[3][1])
    
    mammalcombi_in  = np.intersect1d(mousecombi_in, in_out_phase[4][0])
    mammalcombi_out = np.intersect1d(mousecombi_out, in_out_phase[4][1])
    
    
    mousecombi_6h  = np.intersect1d(in_out_phase_6h[0], in_out_phase_6h[1])
    mousecombi_6h  = np.intersect1d(mousecombi_6h, in_out_phase_6h[2])
    mousecombi_6h  = np.intersect1d(mousecombi_6h, in_out_phase_6h[3])
    
    mammalcombi_6h  = np.intersect1d(mousecombi_6h, in_out_phase_6h[4])
    
    
    arrnegposneu = np.zeros((len(tcgaluadgenes), 565))
    for i, v in enumerate(tcgaluadgenes):
        if len(np.where(tcgaluadgenes == v)[0]) > 0:
            ind = np.where(tcgaluadgenes == v)[0][0]
            arrnegposneu[i] = tcgaluad.iloc[ind].values
        else:
            arrnegposneu[i] *= np.nan
    
    normalarr = []
    for i, v in enumerate(normalfiles):
        if np.any(arrnegposneu.T[v] > 10.):
            print(arrnegposneu.T[v], v)
        else:
            normalarr.append(arrnegposneu.T[v])
    normalarr = np.asanyarray(normalarr).T
    humannormalinout   = [[],[]]
    humannormalinoutrs = [[],[]]
    querygene = normalarr[np.where(tcgaluadgenes == startgene)[0][0]]
    for i, v in enumerate(tcgaluadgenes):
        current  = normalarr[np.where(tcgaluadgenes == v)[0][0]]
        spearman = sc.stats.spearmanr(querygene, current)[0]
        if spearman > 0.6: 
            humannormalinout[0].append(v)
            humannormalinoutrs[0].append(spearman)
        if spearman < -0.6: 
            humannormalinout[1].append(v)
            humannormalinoutrs[1].append(spearman)
    
    #spearman 0.6 is very restrictive (40 genes) for TCGA data
    mammalcombihomo_in = np.intersect1d(mammalcombi_in, humannormalinout[0])
    mammalcombihomo_out = np.intersect1d(mammalcombi_out, humannormalinout[1])
    
    #no downsampling!
    
    tumorarr = []
    for i, v in enumerate(tumorfiles):
        if np.any(arrnegposneu.T[v] > 10.):
            pass
        else:
            tumorarr.append(arrnegposneu.T[v])
    tumorarr = np.asanyarray(tumorarr).T
    humantumorinout   = [[],[]]
    humantumorinoutrs = [[],[]]
    querygene = tumorarr[np.where(tcgaluadgenes == startgene)[0][0]]
    for i, v in enumerate(tcgaluadgenes):
        current  = tumorarr[np.where(tcgaluadgenes == v)[0][0]]
        spearman = sc.stats.spearmanr(querygene, current)[0]
        if spearman > 0.6: 
            humantumorinout[0].append(v)
            humantumorinoutrs[0].append(spearman)
        if spearman < -0.6: 
            humantumorinout[1].append(v)
            humantumorinoutrs[1].append(spearman)
    
    
    #sorted after zhang
    mousecombisorted_in = []
    for i, v in enumerate(in_out_phase[0][0]):
        if (v in mousecombi_in) and (v not in mousecombisorted_in): mousecombisorted_in.append(v)
    mousecombisorted_out = []
    for i, v in enumerate(in_out_phase[0][1]):
        if (v in mousecombi_out) and (v not in mousecombisorted_out): mousecombisorted_out.append(v)
    
    mammalcombisorted_in = []
    for i, v in enumerate(in_out_phase[0][0]):
        if (v in mammalcombi_in) and (v not in mammalcombisorted_in): mammalcombisorted_in.append(v)
    mammalcombisorted_out = []
    for i, v in enumerate(in_out_phase[0][1]):
        if (v in mammalcombi_out) and (v not in mammalcombisorted_out): mammalcombisorted_out.append(v)
        
    
    
    print(np.append(mammalcombisorted_in, mammalcombisorted_out))
        
    #removes genes which are not available in all mammalian gene lists, although they are shared by all mouse studies
    mousecombisorted2 = []
    for i, v in enumerate(np.append(mousecombisorted_in, mousecombisorted_out)):
        if v in allgenes: mousecombisorted2.append(v)
    
    
    
    startgene_i = np.copy(datasets[0][1].iloc[np.where(datasets[0][0] == startgene)[0][0]].values)
    startgene_i -= min(startgene_i)
    startgene_i /= max(startgene_i)
    for j, w in enumerate(pd.unique(in_out_phase[0][0][np.argsort(in_out_phase_rs[0][0])])[::-1][:100][::-1]):
        
        indi = np.where(datasets[0][0] == w)[0][0]
        y    = np.copy(datasets[0][1].iloc[indi].values)
        y -= min(y)
        y /= max(y)
        ax[0].scatter(timel2[0], y, color=cmap(norm(in_out_phase_rs[0][0][np.where(in_out_phase[0][0] == w)[0]][0]), alpha=0.5))
        ax[0].plot(timel2[0], y, color=cmap(norm(in_out_phase_rs[0][0][np.where(in_out_phase[0][0] == w)[0]][0]), alpha=0.5))
        
            
    startgene_i = np.copy(datasets[0][1].iloc[np.where(datasets[0][0] == startgene)[0][0]].values)
    startgene_i -= min(startgene_i)
    startgene_i /= max(startgene_i)
    for j, w in enumerate(pd.unique(in_out_phase[0][1][np.argsort(in_out_phase_rs[0][1])])[:100][::-1]):
            
        indi = np.where(datasets[0][0] == w)[0][0]
        y    = np.copy(datasets[0][1].iloc[indi].values)
        y -= min(y)
        y /= max(y)
        ax[1].scatter(timel2[0], y, color=cmap(norm(in_out_phase_rs[0][1][np.where(in_out_phase[0][1] == w)[0]][0]), alpha=0.5))
        ax[1].plot(timel2[0], y, color=cmap(norm(in_out_phase_rs[0][1][np.where(in_out_phase[0][1] == w)[0]][0]), alpha=0.5))
    
    j = 0
    for k, w in enumerate(pd.unique(in_out_phase_6h[0])[np.arange(len(pd.unique(in_out_phase_6h[0])))<100]):
        indi = np.where(datasets[0][0] == w)[0][0]
        y    = np.copy(datasets[0][1].iloc[indi].values)
       
        y -= min(y)
        y /= max(y)
        ax[2].scatter(timel2[0], y, color=cmap(norm(in_out_phase_6h_rs[0][np.where(in_out_phase_6h[0] == w)[0]][0])), alpha=0.5)
        ax[2].plot(timel2[0], y, color=cmap(norm(in_out_phase_6h_rs[0][np.where(in_out_phase_6h[0] == w)[0]][0])), alpha=0.5)
        
    
    ax[0].set_xticks([18, 30, 42, 54, 66], fontsize=10)
    ax[1].set_xticks([18, 30, 42, 54, 66], fontsize=10)
    ax[2].set_xticks([18, 30, 42, 54, 66], fontsize=10)
    ax[0].set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=10)
    ax[1].set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=10)
    ax[2].set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=10)
    ax[0].set_yticklabels([0., '', '', '', '', 1.])
    ax[1].set_yticklabels([0., '', '', '', '', 1.])
    ax[2].set_yticklabels([0., '', '', '', '', 1.])
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[2].set_facecolor('gray')
    ax[1].set_xlabel('Time (h)', fontsize=12)
    ax[0].set_ylabel('Relative gene expression', fontsize=12)
    ax[1].set_ylabel('Relative gene expression', fontsize=12)
    ax[2].set_ylabel('Relative gene expression', fontsize=12)
    ax[0].set_title(r'$r_{S} \approx 1$', fontsize=12)
    ax[1].set_title(r'$r_{S} \approx -1$', fontsize=12)
    ax[2].set_title(r'$r_{S} \approx 0$', fontsize=12)
    
    


norm0 = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap0 = mpl.colormaps['seismic']


marker = ['o', 's', 'p', 'd', 'X']

startgene = 'PER3'

zhangbased_genelist12 = np.append(pd.unique(in_out_phase[0][0][np.argsort(in_out_phase_rs[0][0])])[::-1][:100][::-1], pd.unique(in_out_phase[0][1][np.argsort(in_out_phase_rs[0][1])])[:100][::-1])
zhangbased_genelist6  = pd.unique(in_out_phase_6h[0])[np.arange(len(pd.unique(in_out_phase_6h[0])))<100]

zhanglung_rs12  = nonrandomrs(zhangbased_genelist12, genes, data, mousetime, startgene=startgene)
rs_lists12 = [zhanglung_rs12]
for i, v in enumerate(zhangbased_genelist12): 
    for j, w in enumerate(rs_lists12):
        if i in w[2]:
            ind = np.arange(len(w[2]))[w[2]==i]
            if w[3][ind] >0.4:
                im = ax2[0].scatter(w[0][ind], w[1][ind], c=zhanglung_rs12[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=200., cmap=cmap0, norm=norm0)
                ax2[1].scatter(w[0][ind], w[3][ind], c=zhanglung_rs12[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=200., cmap=cmap0, norm=norm0)
                ax2[2].scatter(w[0][ind], w[4][ind], c=zhanglung_rs12[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=200., cmap=cmap0, norm=norm0)
                if v in ['PER3', 'ARNTL', 'RORC', 'TEF']: 
                    ax2[0].annotate(v, (w[0][ind], w[1][ind]))
                    ax2[1].annotate(v, (w[0][ind], w[3][ind]))
                    ax2[2].annotate(v, (w[0][ind], w[4][ind]))

  
                
zhanglung_rs6  = nonrandomrs(zhangbased_genelist6, genes, data, mousetime, startgene=startgene)
rs_lists6 = [zhanglung_rs6]
for i, v in enumerate(zhangbased_genelist6): 
    for j, w in enumerate(rs_lists6):
        if i in w[2]:
            ind = np.arange(len(w[2]))[w[2]==i]
            if w[3][ind] >0.4:
                im = ax2[0].scatter(w[0][ind], w[1][ind], c=zhanglung_rs6[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=200., cmap=cmap0, norm=norm0)
                ax2[1].scatter(w[0][ind], w[3][ind], c=zhanglung_rs6[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=200., cmap=cmap0, norm=norm0)
                ax2[2].scatter(w[0][ind], w[4][ind], c=zhanglung_rs6[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=200., cmap=cmap0, norm=norm0)
                if v in ['PER3', 'ARNTL', 'RORC', 'TEF']: 
                    ax2[0].annotate(v, (w[0][ind], w[1][ind]))
                    ax2[1].annotate(v, (w[0][ind], w[3][ind]))
                    ax2[2].annotate(v, (w[0][ind], w[4][ind]))

ax2[0].set_xlabel(r'$r_{S}$', fontsize=12)
ax2[0].set_ylabel('Phase difference to '+startgene, fontsize=12)
ax2[0].set_xlim(-1.1, 1.1)
ax2[0].set_xticks([-1., -0.5, 0., 0.5, 1.])
ax2[0].set_yticks([0., 6., 12., 18., 24.])
ax2[0].set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
ax2[0].set_yticklabels([0, 6, 12, 18, 24], fontsize=10.)
ax2[0].grid()

ax2[1].set_xlabel(r'$r_{S}$', fontsize=12)
ax2[1].set_ylabel(r'$R^{2}$', fontsize=12)
ax2[1].set_xlim(-1.1, 1.1)
ax2[1].set_ylim(-0.1, 1.1)
ax2[1].set_xticks([-1., -0.5, 0., 0.5, 1.])
ax2[1].set_yticks([0., 0.25, 0.5, 0.75, 1.])
ax2[1].set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
ax2[1].set_yticklabels([0., '', 0.5, '', 1.], fontsize=10.)
ax2[1].grid()

ax2[2].set_xlabel(r'$r_{S}$', fontsize=12)
ax2[2].set_ylabel('Gene expression amplitude', fontsize=12)
ax2[2].set_ylim(-0.1, 2.1)
ax2[2].set_yticks([0., 0.5, 1, 1.5, 2])
ax2[2].set_yticklabels([0., '', 1., '', 2.], fontsize=10.)
ax2[2].set_xticks([-1., -0.5, 0., 0.5, 1.])
ax2[2].set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
ax2[2].grid()

#%% save all gene lists as csv

'''

genelist_to_csv(pd.unique(in_out_phase[0][0][np.argsort(in_out_phase_rs[0][0])])[::-1][:100].tolist(),
                pd.unique(in_out_phase[0][1][np.argsort(in_out_phase_rs[0][1])])[:100].tolist(),
                pd.unique(in_out_phase_6h[0])[np.arange(len(pd.unique(in_out_phase_6h[0])))].tolist(),
                'rs_vs_phi_genelist_Zhang')

genelist_to_csv(pd.unique(in_out_phase[1][0][np.argsort(in_out_phase_rs[1][0])])[::-1][:100].tolist(),
                pd.unique(in_out_phase[1][1][np.argsort(in_out_phase_rs[1][1])])[:100].tolist(),
                pd.unique(in_out_phase_6h[1])[np.arange(len(pd.unique(in_out_phase_6h[1])))].tolist(),
                'rs_vs_phi_genelist_Esseryoung')

genelist_to_csv(pd.unique(in_out_phase[2][0][np.argsort(in_out_phase_rs[2][0])])[::-1][:100].tolist(),
                pd.unique(in_out_phase[2][1][np.argsort(in_out_phase_rs[2][1])])[:100].tolist(),
                pd.unique(in_out_phase_6h[2])[np.arange(len(pd.unique(in_out_phase_6h[2])))].tolist(),
                'rs_vs_phi_genelist_Esseraged')

genelist_to_csv(pd.unique(in_out_phase[3][0][np.argsort(in_out_phase_rs[3][0])])[::-1][:100].tolist(),
                pd.unique(in_out_phase[3][1][np.argsort(in_out_phase_rs[3][1])])[:100].tolist(),
                pd.unique(in_out_phase_6h[3])[np.arange(len(pd.unique(in_out_phase_6h[3])))].tolist(),
                'rs_vs_phi_genelist_Esserold')

genelist_to_csv(pd.unique(in_out_phase[4][0][np.argsort(in_out_phase_rs[4][0])])[::-1][:100].tolist(),
                pd.unique(in_out_phase[4][1][np.argsort(in_out_phase_rs[4][1])])[:100].tolist(),
                pd.unique(in_out_phase_6h[4])[np.arange(len(pd.unique(in_out_phase_6h[4])))].tolist(),
                'rs_vs_phi_genelist_Mure')


genelist_to_csv(mousecombi_in.tolist(),
                mousecombi_out.tolist(),
                mousecombi_6h.tolist(),
                'rs_vs_phi_genelist_mousecombi')

genelist_to_csv(mammalcombi_in.tolist(),
                mammalcombi_out.tolist(),
                mammalcombi_6h.tolist(),
                'rs_vs_phi_genelist_mammalcombi')

'''
#%% colorbars for inkscape

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.05])
cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap0, norm=norm0, ticks=[-1., -0.5, 0., 0.5, 1.])
cb.ax.tick_params(labelsize=10.)
cb.set_label(label=r'$r_{S}$ to PER3 for Zhang', fontsize=12.)

#%% top half of Figure 3 for correlation paper

norm0 = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap0 = mpl.colormaps['seismic']

fi, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(15/2.54, 10/2.54))

marker = ['o', 's', 'p', 'd', 'X']
    
mammal_genelist = np.append(mammalcombisorted_in, mammalcombisorted_out)
mammal_genelist = np.append(mammal_genelist, mammalcombi_6h)
zhanglung_rs_mammal  = nonrandomrs(mammal_genelist, genes, data, mousetime, startgene=startgene)
esserlung_rs_mammal  = nonrandomrs(mammal_genelist, esserlungyounggenes, esserlungyoung, mousetime2, startgene=startgene)
esserlung2_rs_mammal = nonrandomrs(mammal_genelist, esserlungagedgenes, esserlungaged, mousetime2, startgene=startgene)
esserlung3_rs_mammal = nonrandomrs(mammal_genelist, esserlungoldgenes, esserlungold, mousetime2, startgene=startgene)
baboon_rs_mammal     = nonrandomrs(mammal_genelist, baboongenes, baboon, baboontime, startgene=startgene)
rs_lists_mammal = [zhanglung_rs_mammal, esserlung_rs_mammal, esserlung2_rs_mammal, esserlung3_rs_mammal, baboon_rs_mammal]


for i, v in enumerate(mammal_genelist): 
    for j, w in enumerate(rs_lists_mammal):
        if i in w[2]:
            ind = np.arange(len(w[2]))[w[2]==i]
            if w[3][ind] >0.4:
                im = ax.scatter(w[0][ind], w[1][ind], c=zhanglung_rs_mammal[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
                if j == 0:
                    ax.annotate(v, (w[0][ind], w[1][ind]))
ax.set_xlim(-1.1, 1.1)
ax.set_xticks([-1., -0.5, 0., 0.5, 1.])
ax.set_yticks([0., 6., 12., 18., 24.])
ax.set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
ax.set_yticklabels([0, 6, 12, 18, 24], fontsize=10.)
ax.grid()
ax.set_ylabel(r'$\Delta \varphi$ to '+startgene, fontsize=12)
ax.set_xlabel(r'$r_{S}$', fontsize=12.)

#%% Figure SI2
norm0 = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap0 = mpl.colormaps['seismic']


fi, ax = plt.subplots(1, 4, figsize=(21/2.54, 4/2.54))
ax = ax.flatten()

marker = ['o', 's', 'p', 'd']

startgene = 'PER3'

zhanglungcolor_rs  = nonrandomrs(np.append(mousecombisorted2, mousecombi_6h), genes, data, mousetime, startgene=startgene)
esserlung1_rs  = nonrandomrs(np.append(mousecombisorted2, mousecombi_6h), esserlungyounggenes, esserlungyoung, mousetime2, startgene=startgene)
esserlung2_rs = nonrandomrs(np.append(mousecombisorted2, mousecombi_6h), esserlungagedgenes, esserlungaged, mousetime2, startgene=startgene)
esserlung3_rs = nonrandomrs(np.append(mousecombisorted2, mousecombi_6h), esserlungoldgenes, esserlungold, mousetime2, startgene=startgene)
rs_lists123 = [zhanglungcolor_rs, esserlung1_rs, esserlung2_rs, esserlung3_rs]

for i, v in enumerate(np.append(mousecombisorted2, mousecombi_6h)): 
    for j, w in enumerate(rs_lists123):
        if i in w[2]:
            ind = np.arange(len(w[2]))[w[2]==i]
            if w[3][ind] >0.4:
                im = ax[j].scatter(w[0][ind], w[1][ind], c=zhanglungcolor_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
                if v in ['PER3', 'ARNTL', 'RORC', 'TEF', 'CRY1']:
                    ax[j].annotate(v, (w[0][ind], w[1][ind]))

ax[0].set_title('Zhang', fontsize=12.)
ax[1].set_title('Wolff (young)', fontsize=12.)
ax[2].set_title('Wolff (aged)', fontsize=12.)
ax[3].set_title('Wolff (old)', fontsize=12.)

for i in range(4):
    ax[i].set_xlim(-1.1, 1.1)
    ax[i].set_xticks([-1., -0.5, 0., 0.5, 1.])
    ax[i].set_yticks([0., 6., 12., 18., 24.])
    ax[i].set_xticklabels([-1., '', 0., '', 1.], fontsize=10.)
    ax[i].set_yticklabels([], fontsize=10.)
    ax[i].grid()
    ax[i].set_xlabel(r'$r_{S}$', fontsize=12.)
ax[0].set_yticklabels([0, '', 12, '', 24], fontsize=10.)
ax[0].set_ylabel(r'$\Delta \phi$ to '+startgene, fontsize=12)

#%% rs vs phi all lung datasets + TCGA, shared mammalian gene list

mammal_genelist = np.append(mammalcombisorted_in, mammalcombisorted_out)
mammal_genelist = np.append(mammal_genelist, mammalcombi_6h)

#use paired samples!
arrnegposneu = np.zeros((len(mammal_genelist), 565))
for i, v in enumerate(mammal_genelist):
    if len(np.where(tcgaluadgenes == v)[0]) > 0:
        ind = np.where(tcgaluadgenes == v)[0][0]
        arrnegposneu[i] = tcgaluad.iloc[ind].values
    else:
        arrnegposneu[i] *= np.nan
        
normalarr = []
for i, v in enumerate(normalmatchedfiles):
    if np.any(arrnegposneu.T[v] > 10.):
        print(arrnegposneu.T[v], v)
    else:
        normalarr.append(arrnegposneu.T[v])
normalarr = np.asanyarray(normalarr)
normrs = np.zeros((len(mammal_genelist), len(mammal_genelist)))
for i in range(len(mammal_genelist)):
    for j in range(len(mammal_genelist)):
        normrs[i][j] = sc.stats.spearmanr(normalarr.T[i], normalarr.T[j])[0]

        
tumorarr = []
for i, v in enumerate(tumormatchedfiles):
    if np.any(arrnegposneu.T[v] > 10.):
        print(arrnegposneu.T[v], v)
    else:
        tumorarr.append(arrnegposneu.T[v])
tumorarr = np.asanyarray(tumorarr)
tumorrs = np.zeros((len(mammal_genelist), len(mammal_genelist)))
for i in range(len(mammal_genelist)):
    for j in range(len(mammal_genelist)):
        tumorrs[i][j] = sc.stats.spearmanr(tumorarr.T[i], tumorarr.T[j])[0]
        
        

norm0 = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap0 = mpl.colormaps['seismic']

marker = ['o', 's', 'p', 'd', 'X']
annotated_genes = ['ARNTL', 'NPAS2', 'PER1', 'PER2', 'PER3', 'RORA', 'RORC', 'NR1D2']


startgene = 'PER3'

zhanglung_rs  = nonrandomrs(mammal_genelist, genes, data, mousetime, startgene=startgene)
esserlung_rs  = nonrandomrs(mammal_genelist, esserlungyounggenes, esserlungyoung, mousetime2, startgene=startgene)
esserlung2_rs = nonrandomrs(mammal_genelist, esserlungagedgenes, esserlungaged, mousetime2, startgene=startgene)
esserlung3_rs = nonrandomrs(mammal_genelist, esserlungoldgenes, esserlungold, mousetime2, startgene=startgene)
baboon_rs     = nonrandomrs(mammal_genelist, baboongenes, baboon, baboontime, startgene=startgene)

rs_lists = [zhanglung_rs, esserlung_rs, esserlung2_rs, esserlung3_rs, baboon_rs]

fi = plt.figure(figsize=(21/2.54, 8/2.54), constrained_layout=True)
gs = GridSpec(1, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])


for i, v in enumerate(mammal_genelist):
    ax1.vlines(normrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax2.vlines(tumorrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax1.annotate(v, (normrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
    ax2.annotate(v, (tumorrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
            

    for j, w in enumerate(rs_lists):
        ind = np.arange(len(w[2]))[w[2]==i]
        if w[3][ind]>0.5:
            im = ax1.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
            im = ax2.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
        

for i, ax in enumerate([ax1, ax2]):
    ax.set_xlabel(r'$r_{S}$ to '+startgene, fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 24.5)
    ax.set_xticks([-1., -0.5, 0., 0.5, 1.])
    ax.set_yticks([0., 6., 12., 18., 24.])
    ax.set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
    ax.grid()

ax1.set_yticklabels([0, 6, 12, 18, 24], fontsize=10.)
ax2.set_yticklabels([], fontsize=10.)
ax1.set_ylabel(r'$\Delta \varphi$ to '+startgene, fontsize=12)
ax1.set_title('LUAD normal')
ax2.set_title('LUAD tumor')


startgene = 'ARNTL'

zhanglung_rs  = nonrandomrs(mammal_genelist, genes, data, mousetime, startgene=startgene)
esserlung_rs  = nonrandomrs(mammal_genelist, esserlungyounggenes, esserlungyoung, mousetime2, startgene=startgene)
esserlung2_rs = nonrandomrs(mammal_genelist, esserlungagedgenes, esserlungaged, mousetime2, startgene=startgene)
esserlung3_rs = nonrandomrs(mammal_genelist, esserlungoldgenes, esserlungold, mousetime2, startgene=startgene)
baboon_rs     = nonrandomrs(mammal_genelist, baboongenes, baboon, baboontime, startgene=startgene)

rs_lists = [zhanglung_rs, esserlung_rs, esserlung2_rs, esserlung3_rs, baboon_rs]

fi = plt.figure(figsize=(21/2.54, 8/2.54), constrained_layout=True)
gs = GridSpec(1, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])


for i, v in enumerate(mammal_genelist):
    ax1.vlines(normrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax2.vlines(tumorrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax1.annotate(v, (normrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
    ax2.annotate(v, (tumorrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
            

    for j, w in enumerate(rs_lists):
        ind = np.arange(len(w[2]))[w[2]==i]
        if w[3][ind]>0.5:
            im = ax1.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
            im = ax2.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
        

for i, ax in enumerate([ax1, ax2]):
    ax.set_xlabel(r'$r_{S}$ to '+startgene, fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 24.5)
    ax.set_xticks([-1., -0.5, 0., 0.5, 1.])
    ax.set_yticks([0., 6., 12., 18., 24.])
    ax.set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
    ax.grid()

ax1.set_yticklabels([0, 6, 12, 18, 24], fontsize=10.)
ax2.set_yticklabels([], fontsize=10.)
ax1.set_ylabel(r'$\Delta \varphi$ to '+startgene, fontsize=12)
ax1.set_title('LUAD normal')
ax2.set_title('LUAD tumor')


startgene = 'NPAS2'

zhanglung_rs  = nonrandomrs(mammal_genelist, genes, data, mousetime, startgene=startgene)
esserlung_rs  = nonrandomrs(mammal_genelist, esserlungyounggenes, esserlungyoung, mousetime2, startgene=startgene)
esserlung2_rs = nonrandomrs(mammal_genelist, esserlungagedgenes, esserlungaged, mousetime2, startgene=startgene)
esserlung3_rs = nonrandomrs(mammal_genelist, esserlungoldgenes, esserlungold, mousetime2, startgene=startgene)
baboon_rs     = nonrandomrs(mammal_genelist, baboongenes, baboon, baboontime, startgene=startgene)

rs_lists = [zhanglung_rs, esserlung_rs, esserlung2_rs, esserlung3_rs, baboon_rs]

fi = plt.figure(figsize=(21/2.54, 8/2.54), constrained_layout=True)
gs = GridSpec(1, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])


for i, v in enumerate(mammal_genelist):
    ax1.vlines(normrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax2.vlines(tumorrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax1.annotate(v, (normrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
    ax2.annotate(v, (tumorrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
            

    for j, w in enumerate(rs_lists):
        ind = np.arange(len(w[2]))[w[2]==i]
        if w[3][ind]>0.5:
            im = ax1.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
            im = ax2.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
        

for i, ax in enumerate([ax1, ax2]):
    ax.set_xlabel(r'$r_{S}$ to '+startgene, fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 24.5)
    ax.set_xticks([-1., -0.5, 0., 0.5, 1.])
    ax.set_yticks([0., 6., 12., 18., 24.])
    ax.set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
    ax.grid()

ax1.set_yticklabels([0, 6, 12, 18, 24], fontsize=10.)
ax2.set_yticklabels([], fontsize=10.)
ax1.set_ylabel(r'$\Delta \varphi$ to '+startgene, fontsize=12)
ax1.set_title('LUAD normal')
ax2.set_title('LUAD tumor')


startgene = 'TEF'

zhanglung_rs  = nonrandomrs(mammal_genelist, genes, data, mousetime, startgene=startgene)
esserlung_rs  = nonrandomrs(mammal_genelist, esserlungyounggenes, esserlungyoung, mousetime2, startgene=startgene)
esserlung2_rs = nonrandomrs(mammal_genelist, esserlungagedgenes, esserlungaged, mousetime2, startgene=startgene)
esserlung3_rs = nonrandomrs(mammal_genelist, esserlungoldgenes, esserlungold, mousetime2, startgene=startgene)
baboon_rs     = nonrandomrs(mammal_genelist, baboongenes, baboon, baboontime, startgene=startgene)

rs_lists = [zhanglung_rs, esserlung_rs, esserlung2_rs, esserlung3_rs, baboon_rs]

fi = plt.figure(figsize=(21/2.54, 8/2.54), constrained_layout=True)
gs = GridSpec(1, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])


for i, v in enumerate(mammal_genelist):
    ax1.vlines(normrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax2.vlines(tumorrs[mammal_genelist==startgene][0][i], 0., 24., color=cmap0(norm0(zhanglung_rs[0][i])), alpha=0.5)
    ax1.annotate(v, (normrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
    ax2.annotate(v, (tumorrs[mammal_genelist==startgene][0][i], 5.2), rotation=90, fontsize=10., alpha=0.8)
            

    for j, w in enumerate(rs_lists):
        ind = np.arange(len(w[2]))[w[2]==i]
        if w[3][ind]>0.5:
            im = ax1.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
            im = ax2.scatter(w[0][ind], w[1][ind], c=zhanglung_rs[0][i], marker=marker[j], edgecolor='gray', alpha=0.8, s=100., cmap=cmap0, norm=norm0)
        

for i, ax in enumerate([ax1, ax2]):
    ax.set_xlabel(r'$r_{S}$ to '+startgene, fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 24.5)
    ax.set_xticks([-1., -0.5, 0., 0.5, 1.])
    ax.set_yticks([0., 6., 12., 18., 24.])
    ax.set_xticklabels([-1., -0.5, 0., 0.5, 1.], fontsize=10.)
    ax.grid()

ax1.set_yticklabels([0, 6, 12, 18, 24], fontsize=10.)
ax2.set_yticklabels([], fontsize=10.)
ax1.set_ylabel(r'$\Delta \varphi$ to '+startgene, fontsize=12)
ax1.set_title('LUAD normal')
ax2.set_title('LUAD tumor')
    
#%% Figure 1 for correlation paper

norm = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap = mpl.colormaps['seismic']

fi = plt.figure(figsize=(25/2.54, 19/2.54), constrained_layout=True)
gs = GridSpec(10, 8, figure=fi)
ax2 = fi.add_subplot(gs[:4, :])
ax3 = fi.add_subplot(gs[4:7, 0:2])
ax4 = fi.add_subplot(gs[4:7, 2:4])
ax5 = fi.add_subplot(gs[7:, 0:2])
ax6 = fi.add_subplot(gs[7:, 2:4])
ax7 = fi.add_subplot(gs[4:, 4:], projection='3d')

per3zhang = np.copy(datasets[0][1].iloc[np.where(datasets[0][0] == 'PER3')[0][0]].values)
arntlzhang = np.copy(datasets[0][1].iloc[np.where(datasets[0][0] == 'ARNTL')[0][0]].values)
rorczhang = np.copy(datasets[0][1].iloc[np.where(datasets[0][0] == 'RORC')[0][0]].values)
tefzhang = np.copy(datasets[0][1].iloc[np.where(datasets[0][0] == 'TEF')[0][0]].values)

rs_phi_gene_r2_amp = nonrandomrs(['PER3', 'ARNTL', 'RORC', 'TEF'], datasets[0][0], datasets[0][1], timel2[0], startgene='PER3')
rs_phi_gene_r2_amp2 = nonrandomrs(['PER3', 'ARNTL', 'RORC', 'TEF'], datasets[0][0], datasets[0][1], timel2[0], startgene='ARNTL')


ax2.axvline(44, color='k')
for i, v in enumerate([(16, 20), (38, 44), (62, 66)]):
     ax2.axvspan(v[0], v[1], color=cmap(norm(rs_phi_gene_r2_amp[0][2])), alpha=0.3)
for i, v in enumerate([(20, 26), (44, 50)]):
     ax2.axvspan(v[0], v[1], color=cmap(norm(rs_phi_gene_r2_amp[0][1])), alpha=0.3)
for i, v in enumerate([(32, 38), (56, 62)]):
     ax2.axvspan(v[0], v[1], color=cmap(norm(rs_phi_gene_r2_amp[0][0])), alpha=0.3)
ax2.plot(timel2[0], per3zhang, color=cmap(norm(rs_phi_gene_r2_amp[0][0])))
ax2.scatter(timel2[0], per3zhang, s=100, color=cmap(norm(rs_phi_gene_r2_amp[0][0])), edgecolor='gray', alpha=0.8, label='PER3')
#ax2.plot(timel[0], tefzhang, color=cmap(norm(rs_phi_gene_r2_amp[0][-1])))
#ax2.scatter(timel[0], tefzhang, s=100, color=cmap(norm(rs_phi_gene_r2_amp[0][-1])), edgecolor='gray', alpha=0.8)
ax2.plot(timel2[0], arntlzhang, color=cmap(norm(rs_phi_gene_r2_amp[0][1])))
ax2.scatter(timel2[0], arntlzhang, s=100, color=cmap(norm(rs_phi_gene_r2_amp[0][1])), edgecolor='gray', alpha=0.8, label='ARNTL')
ax2.plot(timel2[0], rorczhang, color=cmap(norm(rs_phi_gene_r2_amp[0][2])))
ax2.scatter(timel2[0], rorczhang, s=100, color=cmap(norm(rs_phi_gene_r2_amp[0][2])), edgecolor='gray', alpha=0.8, label='RORC')

ax3.plot(per3zhang, arntlzhang, color='gray', alpha=0.8)
ax4.plot(per3zhang, rorczhang, color='gray', alpha=0.8)
ax5.plot(per3zhang, tefzhang, color='gray', alpha=0.8)
ax6.plot(arntlzhang, rorczhang, color='gray', alpha=0.8)
ax3.scatter(per3zhang, arntlzhang, c=np.tile(timel2[0][:12], 2), s=100, edgecolor='gray', alpha=0.8, cmap='twilight')
ax4.scatter(per3zhang, rorczhang,  c=np.tile(timel2[0][:12], 2), s=100, edgecolor='gray', alpha=0.8, cmap='twilight')
ax5.scatter(per3zhang, tefzhang,   c=np.tile(timel2[0][:12], 2), s=100, edgecolor='gray', alpha=0.8, cmap='twilight')
ax6.scatter(arntlzhang, rorczhang, c=np.tile(timel2[0][:12], 2), s=100, edgecolor='gray', alpha=0.8, cmap='twilight')

ax3.set_title(r'$r_{s}$ = '+str(np.round(rs_phi_gene_r2_amp[0][1], 2)), fontsize=12.)
ax4.set_title(r'$r_{s}$ = '+str(np.round(rs_phi_gene_r2_amp[0][2], 2)), fontsize=12.)
ax5.set_title(r'$r_{s}$ = '+str(np.round(rs_phi_gene_r2_amp[0][3], 2)), fontsize=12.)
ax6.set_title(r'$r_{s}$ = '+str(np.round(rs_phi_gene_r2_amp2[0][2], 2)), fontsize=12.)

ax7.plot(per3zhang, arntlzhang, rorczhang, color='gray')
im = ax7.scatter(per3zhang, arntlzhang, rorczhang, s=700, alpha=0.5, c=np.tile(timel2[0][:12], 2), edgecolor='gray', cmap='twilight')


ax2.set_xlim(16, 66)
ax2.set_ylim(5.3, 10.3)
ax2.set_xticks(np.array([20, 26, 32, 38, 44, 50, 56, 62]))
ax2.set_xticklabels(np.array([20, '', 32, '', 44, '', 56, '']), fontsize=10)
ax2.set_xlabel('Time (h)', fontsize=12)
ax2.grid(color='dimgray')
ax2.set_ylabel('log2(Gene expression)', fontsize=12)
ax2.set_yticks(np.array([6, 8, 10]))
ax2.set_yticklabels(np.array([6, 8, 10]), fontsize=10)
ax2.legend(fontsize=10)

ax3.set_xticks(np.array([8, 9.5]))
ax3.set_xticklabels(np.array([8, 9.5]), fontsize=10)
ax3.set_yticks(np.array([6, 8.4]))
ax3.set_yticklabels(np.array([6, 8.4]), fontsize=10)
ax4.set_xticks(np.array([8, 9.5]))
ax4.set_xticklabels(np.array([8, 9.5]), fontsize=10)
ax4.set_yticks(np.array([8.1, 9.5]))
ax4.set_yticklabels(np.array([8.1, 9.5]), fontsize=10)
ax5.set_xticks(np.array([8, 9.5]))
ax5.set_xticklabels(np.array([8, 9.5]), fontsize=10)
ax5.set_yticks(np.array([8.5, 9.7]))
ax5.set_yticklabels(np.array([8.5, 9.7]), fontsize=10)
ax6.set_xticks(np.array([6.1, 8.4]))
ax6.set_xticklabels(np.array([6.1, 8.4]), fontsize=10)
ax6.set_yticks(np.array([8.1, 9.5]))
ax6.set_yticklabels(np.array([8.1, 9.5]), fontsize=10)
ax3.set_xlabel('PER3', fontsize=12)
ax3.set_ylabel('ARNTL', fontsize=12)
ax4.set_xlabel('PER3', fontsize=12)
ax4.set_ylabel('RORC', fontsize=12)
ax5.set_xlabel('PER3', fontsize=12)
ax5.set_ylabel('TEF', fontsize=12)
ax6.set_xlabel('ARNTL', fontsize=12)
ax6.set_ylabel('RORC', fontsize=12)


ax7.set_xlabel('PER3', fontsize=12)
ax7.set_ylabel('ARNTL', fontsize=12)
ax7.set_zlabel('RORC', fontsize=12, rotation=90)
ax7.set_xticks(np.array([8, 9.5]))
ax7.set_xticklabels(np.array([8, 9.5]), fontsize=10)
ax7.set_yticks(np.array([6, 8.4]))
ax7.set_yticklabels(np.array([6, 8.4]), fontsize=10)
ax7.set_zticks(np.array([8.1, 9.5]))
ax7.set_zticklabels(np.array([8.1, 9.5]), fontsize=10)
ax7.view_init(25, 50)
cb  = fi.colorbar(im,  ax=ax7, ticks=[20, 38])
cb.ax.tick_params(labelsize=10.)
cb.ax.set_yticklabels(['20|44', '38|62'])
cb.set_label(label='Time (h)', fontsize=12.)


#%% correlationmatrix

norm = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap = mpl.colormaps['bwr']
cmap.set_bad('gray')


fi = plt.figure(figsize=(21/2.54, 29.7/2.54), constrained_layout=True)
gs = GridSpec(8, 2, figure=fi)

ax = []
for i in range(2):
    axis = []
    for k in range(7):
        axi = fi.add_subplot(gs[k+1:k+2, i:i+1])
        axis.append(axi)
    ax.append(axis)
ax = np.asanyarray(ax).T



queries = [np.append(np.append(np.sort(mousecombisorted_in), np.sort(mousecombisorted_out)), np.sort(mousecombi_6h)), np.append(np.append(mammalcombisorted_in, mammalcombisorted_out), mammalcombi_6h)]
querienames = ['Mouse +-12h \n and +-6h genes', 'Mammalian +-12h \n and +-6h genes', '36 random genes']
datasetnames = ['zhang', 'esser_young', 'esser_aged', 'esser_old', 'baboon', 'LUAD_normal', 'LUAD_cancer']



norm = MidpointNormalize(vmin=-1.,   vmax=1., midpoint=0.)
cmap = mpl.colormaps['bwr']
cmap.set_bad('gray')
corrcorr = [[],[],[],[]]
for k, l in enumerate(queries):
    for i, v in enumerate(datasets):
        corr = correlationmatrix(l, v[0], v[1])
        corrcorr[k].append(corr)
    
        ax[i][k].imshow(corr, cmap=cmap, norm=norm)
        ax[i][k].set_xticks([], rotation=90, fontsize=10)
        ax[i][k].set_yticks([], fontsize=10)
        
        if k == 0: ax[i][k].set_ylabel(datasetnames[i], fontsize=10)


for k, l in enumerate(queries):
    corr = correlationmatrix_tcga_paired(l, tcgaluadgenes, tcgaluad, normalfiles, tumorfiles, normalmatchedfiles, tumormatchedfiles)
    corrcorr[k].append(corr[0])
    corrcorr[k].append(corr[1])

    ax[5][k].imshow(corr[0], cmap=cmap, norm=norm)
    ax[5][k].set_xticks([], rotation=90, fontsize=10)
    ax[5][k].set_yticks([], fontsize=10)
    
    ax[6][k].imshow(corr[1], cmap=cmap, norm=norm)
    
    ax[0][k].set_title(querienames[k], fontsize=10)

ax[-2][0].set_ylabel(datasetnames[-2], fontsize=10)
ax[-1][0].set_ylabel(datasetnames[-1], fontsize=10)

fsizes = [4., 4.]
for k in range(2):
    if k != 0:
        ax[0][k].set_xticks(np.arange(len(queries[k])))
        ax[0][k].set_yticks(np.arange(len(queries[k])))
    
        ax[0][k].set_xticklabels(queries[k], rotation=90, fontsize=fsizes[k])
        ax[0][k].set_yticklabels(queries[k], fontsize=fsizes[k])
        
    else:
    
        ax[0][k].set_xticks([30, 46, 60, 106])
        ax[0][k].set_yticks([30, 46, 60, 106])
    
        ax[0][k].set_xticklabels(['PER3', 'TEF', 'ARNTL', 'RORC'], rotation=90, fontsize=fsizes[k])
        ax[0][k].set_yticklabels(['PER3', 'TEF', 'ARNTL', 'RORC'], fontsize=fsizes[k])

#%% colorbar for rs

norm = MidpointNormalize(vmin=-1., vmax=1., midpoint=0.)
cmap = mpl.cm.get_cmap('bwr')
fig = plt.figure(figsize=(30., 4))
ax = fig.add_axes([0.05, 0.80, 0.9, 0.05])
cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap, norm=norm, ticks=[-1., -0.5, 0., 0.5, 1.])
cb.ax.tick_params(labelsize=18.)
cb.set_label(label=r'$r_{S}$', fontsize=22.)
plt.show()

#%% cosine similarity
queries = [np.append(mousecombisorted2, mousecombi_6h), np.append(np.append(mammalcombisorted_in, mammalcombisorted_out), mammalcombi_6h)]
datasetnames = ['Zhang', 'Wolff (young)', 'Wolff (aged)', 'Wolff (old)', 'Mure', 'LUAD normal', 'LUAD cancer']
querienames = ['Mouse \n +-12h and +-6h genes', 'Mammalian \n +-12h and +-6h genes']

norm = MidpointNormalize(vmin=0.,   vmax=1., midpoint=0.5)

fig, ax = plt.subplots(1, 2, figsize=(21/2.54, 10.5/2.54))
ax = ax.flatten()



for k, l in enumerate(queries):
    cosdist = np.zeros((len(datasetnames), len(datasetnames)))
    for i in range(len(corrcorr[k])):
        for j in range(len(corrcorr[k])):
            x = corrcorr[k][i].flatten()
            y = corrcorr[k][j].flatten()
            x[np.isnan(y)] = np.nan
            y[np.isnan(x)] = np.nan
            cosdist[i][j] = np.nansum(x*y)/(np.sqrt(np.nansum(x**2.))*np.sqrt(np.nansum(y**2.)))
            
            
    cosdisttriu = np.triu(cosdist)
    cosdisttriu[cosdisttriu==0] = np.nan
    ax[k].imshow(cosdisttriu, cmap='Purples', norm=norm)
            
    for i, v in enumerate(cosdist):
        for j, w in enumerate(cosdist):
            if np.isnan(cosdisttriu[i,j]) != True:
                if np.round(cosdist[i,j], 2) > 0.5:
                    text = ax[k].text(j, i, np.round(cosdist[i,j], 2), ha="center", va="center", color="white", fontsize=6)
                else:
                    text = ax[k].text(j, i, np.round(cosdist[i,j], 2), ha="center", va="center", color='k', fontsize=6)
        
    ax[k].set_title(querienames[k], fontsize=10)
    

#%% PCA, esser's data is put together to get a larger population of mice

norm = MidpointNormalize(vmin=0., vmax=24., midpoint=12.)

rng = np.random.default_rng()
randominidices = np.arange(len(np.tile(mousetime2, 3)))
rng.shuffle(randominidices)


query = np.append(np.append(mammalcombisorted_in, mammalcombisorted_out), mammalcombi_6h)

x1 = querygenearray(query, esserlungyounggenes, esserlungyoung)
x12 = querygenearray(query, esserlungagedgenes, esserlungaged)
x13 = querygenearray(query, esserlungoldgenes, esserlungold)


esserlunggenes = [esserlungyounggenes, esserlungagedgenes, esserlungoldgenes]

per3  = np.array([])
arntl = np.array([])
for i, v in enumerate([esserlungyoung, esserlungaged, esserlungold]):
    per3  = np.append(per3, v.iloc[np.where(esserlunggenes[i] == 'PER3')[0][0]].values)
    arntl = np.append(arntl, v.iloc[np.where(esserlunggenes[i] == 'ARNTL')[0][0]].values)
per3 = per3[randominidices]
arntl = arntl[randominidices]


x3 = np.append(x1, x12, axis=1)
x3 = np.append(x3, x13, axis=1)


x3 = x3.T[randominidices].T

mousetime3 = np.tile(mousetime2, 3)[randominidices]

times = [mousetime3]
marker = np.repeat(np.array(['s', 'p', 'd']), 12)[randominidices]
ctime = np.tile(np.arange(0, 24, 4),6)[randominidices]


fi = plt.figure(figsize=(25/2.54, 25/2.54), constrained_layout=True)
gs = GridSpec(3, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])
ax3 = fi.add_subplot(gs[1:2, :1])
ax4 = fi.add_subplot(gs[1:2, 1:])
ax = [ax1, ax2, ax3, ax4]

for i, x in enumerate([x3]):
    pca = decomposition.PCA(n_components=2)
    X_r = pca.fit(x.T).transform(x.T)
    pcaphase  = pcaphases(X_r)
    pcaphase[pcaphase <0.] += 2.*np.pi
    pcaphase = 24.*(pcaphase/(2.*np.pi))
    
    for n, m in enumerate(marker):
        ax[0].scatter(X_r.T[0][n], X_r.T[1][n], c=ctime[n], s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    for n, m in enumerate(marker):
        ax[1].scatter(times[i][n], pcaphase[n], c=ctime[n], s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    popt, pcov = curve_fit(oscifunc, pcaphase, arntl)
    ax[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='k')
    popt2, pcov2 = curve_fit(oscifunc, pcaphase, per3)
    ax[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    for n, m in enumerate(marker):
        ax[2].scatter(pcaphase[n], arntl[n], c=ctime[n], s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    
    popt, pcov = curve_fit(oscifunc, times[i], arntl)
    ax[3].plot(np.arange(18, 62, 0.1), oscifunc(np.arange(18, 62, 0.1), *popt), alpha=0.3, color='k')
    popt2, pcov2 = curve_fit(oscifunc, times[i], per3)
    ax[3].plot(np.arange(18, 62, 0.1), oscifunc(np.arange(18, 62, 0.1), *popt2), alpha=0.3, color='r')
    for n, m in enumerate(marker):
        ax[3].scatter(times[i][n], arntl[n], c=ctime[n], s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    
ax[0].set_xticks([0])
ax[0].set_xticklabels([0], fontsize=10)
ax[0].set_yticks([0])
ax[0].set_yticklabels([0], fontsize=10)
ax[0].axvline(0, color='gray', alpha=0.5)
ax[0].axhline(0, color='gray', alpha=0.5)
ax[0].set_xlabel('PCA1', fontsize=12)
ax[0].set_ylabel('PCA2', fontsize=12)
ax[1].set_xticks([18, 30, 42, 54, 66])
ax[1].set_xticklabels([0, 12, 24, 36, 48], fontsize=10)
ax[1].set_ylabel('phase by PCA (h)', fontsize=12)
ax[1].set_xlabel('recorded time (h)', fontsize=12)
ax[1].set_yticks([0, 12, 24])
ax[1].set_yticklabels([0, 12, 24], fontsize=10)
ax[1].grid()
ax[2].set_xlabel('phase by PCA (h)', fontsize=12)
ax[2].set_xticks([0, 4, 8, 12, 16, 20, 24])
ax[2].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
ax[2].grid()
ax[3].set_xlabel('recorded time (h)', fontsize=12)
ax[3].set_xticks(np.arange(18, 66, 4))
ax[3].set_xticklabels([0, '', '', 12, '', '', 24, '', '', 36, '', ''], fontsize=10)
ax[3].grid()
ax[2].set_ylim(1.1, 7.1)
ax[3].set_ylim(1.1, 7.1)

#%% PCA Zhang

norm = MidpointNormalize(vmin=0., vmax=24., midpoint=12.)

rng = np.random.default_rng()
randominidices = np.arange(len(mousetime))
rng.shuffle(randominidices)


query = np.append(np.append(zhangbased_genelist12[100 - 1:100], zhangbased_genelist12[-1:]), zhangbased_genelist6)

x1 = querygenearray(query, genes, data)



per3  = data.iloc[np.where(genes == 'PER3')[0][0]].values
arntl = data.iloc[np.where(genes == 'ARNTL')[0][0]].values
per3  = per3[randominidices]
arntl = arntl[randominidices]


x1 = x1.T[randominidices].T

mousetime3 = mousetime[randominidices]

times = mousetime3
m = 'o'
ctime = np.tile(np.arange(0, 24, 2),2)[randominidices]

fi = plt.figure(figsize=(25/2.54, 25/2.54), constrained_layout=True)
gs = GridSpec(3, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])
ax3 = fi.add_subplot(gs[1:2, :1])
ax4 = fi.add_subplot(gs[1:2, 1:])
ax = [ax1, ax2, ax3, ax4]

for i, x in enumerate([x1]):

    pca = decomposition.PCA(n_components=2)
    X_r = pca.fit(x.T).transform(x.T)
    pcaphase  = pcaphases(X_r)
    pcaphase[pcaphase <0.] += 2.*np.pi
    pcaphase = 24.*(pcaphase/(2.*np.pi))
    
    ax[0].scatter(X_r.T[0], X_r.T[1], c=ctime, s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    ax[1].scatter(times, pcaphase, c=ctime, s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    popt, pcov = curve_fit(oscifunc, pcaphase, arntl)
    ax[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='k')
    popt2, pcov2 = curve_fit(oscifunc, pcaphase, per3)
    ax[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[2].scatter(pcaphase, arntl, c=ctime, s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    
    popt, pcov = curve_fit(oscifunc, times, arntl)
    ax[3].plot(np.arange(18, 62, 0.1), oscifunc(np.arange(18, 62, 0.1), *popt), alpha=0.3, color='k')
    popt2, pcov2 = curve_fit(oscifunc, times, per3)
    ax[3].plot(np.arange(18, 62, 0.1), oscifunc(np.arange(18, 62, 0.1), *popt2), alpha=0.3, color='r')
    ax[3].scatter(times, arntl, c=ctime, s=100, alpha=0.8, edgecolor='k', marker=m, norm=norm, cmap='twilight')
    
    
    ax[0].set_xticks([0])
    ax[0].set_xticklabels([0], fontsize=10)
    ax[0].set_yticks([0])
    ax[0].set_yticklabels([0], fontsize=10)
    ax[0].axvline(0, color='gray', alpha=0.5)
    ax[0].axhline(0, color='gray', alpha=0.5)
    ax[0].set_xlabel('PCA1', fontsize=12)
    ax[0].set_ylabel('PCA2', fontsize=12)
    ax[1].set_xticks([18, 30, 42, 54, 66])
    ax[1].set_xticklabels([0, 12, 24, 36, 48], fontsize=10)
    ax[1].set_ylabel('phase by PCA (h)', fontsize=12)
    ax[1].set_xlabel('recorded time (h)', fontsize=12)
    ax[1].set_yticks([0, 12, 24])
    ax[1].set_yticklabels([0, 12, 24], fontsize=10)
    ax[1].grid()
    ax[2].set_xlabel('phase by PCA (h)', fontsize=12)
    ax[2].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[2].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[2].grid()
    ax[3].set_xlabel('recorded time (h)', fontsize=12)
    ax[3].set_xticks(np.arange(18, 66, 4))
    ax[3].set_xticklabels([0, '', '', 12, '', '', 24, '', '', 36, '', ''], fontsize=10)
    ax[3].grid()

#%% PCA Mure

norm = MidpointNormalize(vmin=0., vmax=24., midpoint=12.)

rng = np.random.default_rng()
randominidices = np.arange(len(baboontime))
rng.shuffle(randominidices)


query = np.append(np.append(mammalcombisorted_in, mammalcombisorted_out), mammalcombi_6h)

x1 = querygenearray(query, baboongenes, baboon)



per3  = data.iloc[np.where(genes == 'PER3')[0][0]].values
arntl = data.iloc[np.where(genes == 'ARNTL')[0][0]].values
per3  = per3[randominidices]
arntl = arntl[randominidices]


x1 = x1.T[randominidices].T

baboontime3 = baboontime[randominidices]

times = baboontime3
m = 'o'
ctime = np.tile(np.arange(0, 24, 2),2)[randominidices]

fi = plt.figure(figsize=(25/2.54, 25/2.54), constrained_layout=True)
gs = GridSpec(3, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])
ax3 = fi.add_subplot(gs[1:2, :1])
ax4 = fi.add_subplot(gs[1:2, 1:])
ax = [ax1, ax2, ax3, ax4]

for i, x in enumerate([x1]):

    pca = decomposition.PCA(n_components=2)
    X_r = pca.fit(x.T).transform(x.T)
    pcaphase  = pcaphases(X_r)
    pcaphase[pcaphase <0.] += 2.*np.pi
    pcaphase = 24.*(pcaphase/(2.*np.pi))
    
    ax[0].scatter(X_r.T[0], X_r.T[1], c=ctime, s=100, alpha=0.8, edgecolor='k', marker='X', norm=norm, cmap='twilight')
    
    ax[1].scatter(times, pcaphase, c=ctime, s=100, alpha=0.8, edgecolor='k', marker='X', norm=norm, cmap='twilight')
    
    popt, pcov = curve_fit(oscifunc, pcaphase, arntl)
    ax[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='k')
    popt2, pcov2 = curve_fit(oscifunc, pcaphase, per3)
    ax[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[2].scatter(pcaphase, arntl, c=ctime, s=100, alpha=0.8, edgecolor='k', marker='X', norm=norm, cmap='twilight')
    
    
    popt, pcov = curve_fit(oscifunc, times, arntl)
    ax[3].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='k')
    popt2, pcov2 = curve_fit(oscifunc, times, per3)
    ax[3].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[3].scatter(times, arntl, c=ctime, s=100, alpha=0.8, edgecolor='k', marker='X', norm=norm, cmap='twilight')
    
    
    ax[0].set_xticks([0])
    ax[0].set_xticklabels([0], fontsize=10)
    ax[0].set_yticks([0])
    ax[0].set_yticklabels([0], fontsize=10)
    ax[0].axvline(0, color='gray', alpha=0.5)
    ax[0].axhline(0, color='gray', alpha=0.5)
    ax[0].set_xlabel('PCA1', fontsize=12)
    ax[0].set_ylabel('PCA2', fontsize=12)
    ax[1].set_xticks([0, 6, 12, 18])
    ax[1].set_xticklabels([0, 6, 12, 18], fontsize=10)
    ax[1].set_ylabel('phase by PCA (h)', fontsize=12)
    ax[1].set_xlabel('recorded time (h)', fontsize=12)
    ax[1].set_xticks([0, 6, 12, 18, 24])
    ax[1].set_xticklabels([0, '', 12, '', 24], fontsize=10)
    ax[1].set_yticks([0, 12, 24])
    ax[1].set_yticklabels([0, 12, 24], fontsize=10)
    ax[1].grid()
    ax[2].set_xlabel('phase by PCA (h)', fontsize=12)
    ax[2].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[2].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[2].grid()
    ax[3].set_xlabel('recorded time (h)', fontsize=12)
    ax[3].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[3].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[3].grid()
    
#%% PCA TCGA

norm = MidpointNormalize(vmin=0., vmax=24., midpoint=12.)


query = np.append(np.append(mammalcombisorted_in, mammalcombisorted_out), mammalcombi_6h)

arrnegposneu = np.zeros((len(query), 565))
for i, v in enumerate(query):
    if len(np.where(tcgaluadgenes == v)[0]) > 0:
        ind = np.where(tcgaluadgenes == v)[0][0]
        arrnegposneu[i] = tcgaluad.iloc[ind].values
    else:
        arrnegposneu[i] *= np.nan
normalarr = []
for i, v in enumerate(normalmatchedfiles):
    if np.any(arrnegposneu.T[v] > 10.):
        print(arrnegposneu.T[v], v)
    else:
        normalarr.append(arrnegposneu.T[v])
        
x1 = np.asanyarray(normalarr, dtype='float64').T
x1[np.isnan(x1)] = 0.

arrnegposneu_perarntlror = np.zeros((3, 565))
for i, v in enumerate(['PER3', 'ARNTL', 'RORC']):
    if len(np.where(tcgaluadgenes == v)[0]) > 0:
        ind = np.where(tcgaluadgenes == v)[0][0]
        arrnegposneu_perarntlror[i] = tcgaluad.iloc[ind].values
    else:
        arrnegposneu_perarntlror[i] *= np.nan
perarntlror = []
for i, v in enumerate(normalmatchedfiles):
    if np.any(arrnegposneu_perarntlror.T[v] > 10.):
        print(arrnegposneu_perarntlror.T[v], v)
    else:
        perarntlror.append(arrnegposneu_perarntlror.T[v])
perarntlror = np.asanyarray(perarntlror)
per3  = perarntlror.T[0]
arntl = perarntlror.T[1]
rorc  = perarntlror.T[2]



perarntlror2 = []
for i, v in enumerate(tumormatchedfiles):
    if np.any(arrnegposneu_perarntlror.T[v] > 10.):
        print(arrnegposneu_perarntlror.T[v], v)
    else:
        perarntlror2.append(arrnegposneu_perarntlror.T[v])
perarntlror2 = np.asanyarray(perarntlror2)
tumorper3  = perarntlror2.T[0]
tumorarntl = perarntlror2.T[1]
tumorrorc  = perarntlror2.T[2]


#some patients have multiple matching tumor samples
normalmatchinginds = []
for i, v in enumerate(tumormatchedcaseids):
    normalmatchinginds.append(np.arange(len(normalmatchedcaseids))[normalmatchedcaseids==v][0])
normalmatchinginds = np.asanyarray(normalmatchinginds)

m = 'o'

fi = plt.figure(figsize=(21/2.54, 29.7/2.54), constrained_layout=True)
gs = GridSpec(4, 2, figure=fi)
ax1 = fi.add_subplot(gs[:1, :1])
ax2 = fi.add_subplot(gs[:1, 1:])
ax3 = fi.add_subplot(gs[1:2, :1])
ax4 = fi.add_subplot(gs[1:2, 1:])
ax5 = fi.add_subplot(gs[2:3, :1])
ax6 = fi.add_subplot(gs[2:3, 1:])
ax = [ax1, ax2, ax3, ax4, ax5, ax6]

for i, x in enumerate([x1]):

    pca = decomposition.PCA(n_components=2)
    X_r = pca.fit(x.T).transform(x.T)
    pcaphase  = pcaphases(X_r)
    pcaphase[pcaphase <0.] += 2.*np.pi
    pcaphase = 24.*(pcaphase/(2.*np.pi))
    
    perrorphase = np.arctan2(per3-np.mean(per3), rorc-np.mean(rorc))
    perrorphase[perrorphase <0.] += 2.*np.pi
    perrorphase = 24.*(perrorphase/(2.*np.pi))
    
    pcaphase3    = []
    perrorphase3 = []
    for i, v in enumerate(normalmatchinginds):
        pcaphase3.append(pcaphase[v])
        perrorphase3.append(perrorphase[v])
    pcaphase3    = np.asanyarray(pcaphase3)
    perrorphase3 = np.asanyarray(perrorphase3)
    
    
    popt, pcov = curve_fit(oscifunc, pcaphase, arntl)
    popt2, pcov2 = curve_fit(oscifunc, pcaphase, per3)
    popt3, pcov3 = curve_fit(oscifunc, pcaphase, rorc)
    

    ax[0].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='b')
    ax[0].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[0].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt3), alpha=0.3, color='g')
    ax[0].scatter(pcaphase, per3, c=perrorphase, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')
    
    ax[2].scatter(per3, rorc, c=perrorphase, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')
    ax[2].plot(oscifunc(np.arange(0, 24, 0.1), *popt2), oscifunc(np.arange(0, 24, 0.1), *popt3), alpha=0.3, color='g')
    
    ax[4].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='b')
    ax[4].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[4].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt3), alpha=0.3, color='g')
    ax[4].scatter(pcaphase, arntl, c=perrorphase, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')


    popt, pcov = curve_fit(oscifunc, pcaphase3, tumorarntl)
    popt2, pcov2 = curve_fit(oscifunc, pcaphase3, tumorper3)
    popt3, pcov3 = curve_fit(oscifunc, pcaphase3, tumorrorc)

    ax[1].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='b')
    ax[1].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[1].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt3), alpha=0.3, color='g')
    ax[1].scatter(pcaphase3, tumorper3, c=perrorphase3, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')
    

    ax[3].scatter(tumorper3, tumorrorc, c=perrorphase3, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')
    ax[3].plot(oscifunc(np.arange(0, 24, 0.1), *popt2), oscifunc(np.arange(0, 24, 0.1), *popt3), alpha=0.3, color='g')

    ax[5].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt), alpha=0.3, color='b')
    ax[5].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2), alpha=0.3, color='r')
    ax[5].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt3), alpha=0.3, color='g')
    ax[5].scatter(pcaphase3, tumorarntl, c=perrorphase3, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')
    
    ax[0].set_ylim(-2.2, 3.2)
    ax[0].set_xlabel('phase by PCA (h)', fontsize=12)
    ax[0].set_ylabel('PER3', fontsize=12.)
    ax[0].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[0].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[0].grid()
    ax[1].set_ylim(-2.2, 3.2)
    ax[1].set_xlabel('phase by PCA (h)', fontsize=12)
    ax[1].set_ylabel('PER3', fontsize=12.)
    ax[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[1].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[1].grid()
    ax[2].set_ylabel('RORC', fontsize=12)
    ax[2].set_xlabel('PER3', fontsize=12)
    ax[2].grid()
    ax[3].set_ylabel('RORC', fontsize=12)
    ax[3].set_xlabel('PER3', fontsize=12)
    ax[3].grid()
    ax[4].set_ylim(-2.2, 1.6)
    ax[4].set_xlabel('phase by PCA (h)', fontsize=12)
    ax[4].set_ylabel('ARNTL', fontsize=12)
    ax[4].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[4].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[4].grid()
    ax[5].set_ylim(-2.2, 1.6)
    ax[5].set_xlabel('phase by PCA (h)', fontsize=12)
    ax[5].set_ylabel('ARNTL', fontsize=12)
    ax[5].set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax[5].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
    ax[5].grid()
    
ax[0].set_title('normal', fontsize=12)
ax[1].set_title('tumor', fontsize=12)



#%% SI figure 6

norm = MidpointNormalize(vmin=0., vmax=24., midpoint=12.)


query = np.append(np.append(mammalcombisorted_in, mammalcombisorted_out), mammalcombi_6h)

arrnegposneu = np.zeros((len(query), 565))
for i, v in enumerate(query):
    if len(np.where(tcgaluadgenes == v)[0]) > 0:
        ind = np.where(tcgaluadgenes == v)[0][0]
        arrnegposneu[i] = tcgaluad.iloc[ind].values
    else:
        arrnegposneu[i] *= np.nan
normalarr = []
for i, v in enumerate(normalmatchedfiles):
    if np.any(arrnegposneu.T[v] > 10.):
        print(arrnegposneu.T[v], v)
    else:
        normalarr.append(arrnegposneu.T[v])
        
x1 = np.asanyarray(normalarr, dtype='float64').T
x1[np.isnan(x1)] = 0.

arrnegposneu_perarntlror = np.zeros((3, 565))
for i, v in enumerate(['PER3', 'ARNTL', 'RORC']):
    if len(np.where(tcgaluadgenes == v)[0]) > 0:
        ind = np.where(tcgaluadgenes == v)[0][0]
        arrnegposneu_perarntlror[i] = tcgaluad.iloc[ind].values
    else:
        arrnegposneu_perarntlror[i] *= np.nan
perarntlror = []
for i, v in enumerate(normalmatchedfiles):
    if np.any(arrnegposneu_perarntlror.T[v] > 10.):
        print(arrnegposneu_perarntlror.T[v], v)
    else:
        perarntlror.append(arrnegposneu_perarntlror.T[v])
perarntlror = np.asanyarray(perarntlror)
per3  = perarntlror.T[0]
arntl = perarntlror.T[1]
rorc  = perarntlror.T[2]



tumorarr = []
#for i, v in enumerate(tumorfiles):
for i, v in enumerate(tumormatchedfiles):
    if np.any(arrnegposneu.T[v] > 10.):
        print(arrnegposneu.T[v], v)
    else:
        tumorarr.append(arrnegposneu.T[v])

perarntlror2 = []
for i, v in enumerate(tumormatchedfiles):
    if np.any(arrnegposneu_perarntlror.T[v] > 10.):
        print(arrnegposneu_perarntlror.T[v], v)
    else:
        perarntlror2.append(arrnegposneu_perarntlror.T[v])
perarntlror2 = np.asanyarray(perarntlror2)
tumorper3  = perarntlror2.T[0]
tumorarntl = perarntlror2.T[1]
tumorrorc  = perarntlror2.T[2]


m = 'o'

fi2 = plt.figure(figsize=(21/2.54, 26/2.54), constrained_layout=True)
gs = GridSpec(6, 2, figure=fi2)
ax8 = fi2.add_subplot(gs[:1, :1])
ax9 = fi2.add_subplot(gs[:1, 1:])
ax10 = fi2.add_subplot(gs[1:2, :1])
ax11 = fi2.add_subplot(gs[1:2, 1:])
ax12 = fi2.add_subplot(gs[2:3, :1])
ax13 = fi2.add_subplot(gs[2:3, 1:])
axx = [ax8, ax9, ax10, ax11, ax12, ax13]

x = x1

pca = decomposition.PCA(n_components=2)
X_r = pca.fit(x.T).transform(x.T)
pcaphase  = pcaphases(X_r)
pcaphase[pcaphase <0.] += 2.*np.pi
pcaphase = 24.*(pcaphase/(2.*np.pi))

pcaphase3 = np.zeros(len(tumormatchedcaseids_index))
for i, v in enumerate(tumormatchedcaseids_index):
    pcaphase3[i] = pcaphase[int(v)]

perrorphase = np.arctan2(per3-np.mean(per3), rorc-np.mean(rorc))
perrorphase[perrorphase <0.] += 2.*np.pi
perrorphase = 24.*(perrorphase/(2.*np.pi))

perrorphase3 = np.zeros(len(tumormatchedcaseids_index))
for i, v in enumerate(tumormatchedcaseids_index):
    perrorphase3[i] = perrorphase[int(v)]

poptn, pcovn = curve_fit(oscifunc, pcaphase, arntl)
popt2n, pcov2n = curve_fit(oscifunc, pcaphase, per3)
popt3n, pcov3n = curve_fit(oscifunc, pcaphase, rorc)

poptt, pcovt = curve_fit(oscifunc, pcaphase3, tumorarntl)
popt2t, pcov2t = curve_fit(oscifunc, pcaphase3, tumorper3)
popt3t, pcov3t = curve_fit(oscifunc, pcaphase3, tumorrorc)


    
axx[0].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *poptn), alpha=0.3, color='k')
axx[0].scatter(pcaphase, arntl, c=perrorphase, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')

axx[1].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *poptt), alpha=0.3, color='k')
axx[1].scatter(pcaphase3, tumorarntl, c=perrorphase3, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')

axx[2].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt3n), alpha=0.3, color='g')
axx[2].scatter(pcaphase, rorc, c=perrorphase, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')

axx[3].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt3t), alpha=0.3, color='g')
axx[3].scatter(pcaphase3, tumorrorc, c=perrorphase3, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')

axx[4].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2n), alpha=0.3, color='r')
axx[4].scatter(pcaphase, per3, c=perrorphase, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')

axx[5].plot(np.arange(0, 24, 0.1), oscifunc(np.arange(0, 24, 0.1), *popt2t), alpha=0.3, color='r')
axx[5].scatter(pcaphase3, tumorper3, c=perrorphase3, s=100, alpha=0.8, edgecolor='k', marker=m, cmap='twilight')


axx[0].set_ylim(-2.5, 2.4)
axx[0].set_xlabel('phase by PCA (h)', fontsize=12)
axx[0].set_ylabel('ARNTL', fontsize=12.)
axx[0].set_xticks([0, 4, 8, 12, 16, 20, 24])
axx[0].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
axx[0].grid()
axx[1].set_ylim(-2.5, 2.4)
axx[1].set_xlabel('phase by PCA (h)', fontsize=12)
axx[1].set_ylabel('ARNTL', fontsize=12.)
axx[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
axx[1].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
axx[1].grid()
axx[2].set_ylim(-2.5, 2.4)
axx[2].set_xlabel('phase by PCA (h)', fontsize=12)
axx[2].set_ylabel('RORC', fontsize=12.)
axx[2].set_xticks([0, 4, 8, 12, 16, 20, 24])
axx[2].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
axx[2].grid()
axx[3].set_ylim(-2.5, 2.4)
axx[3].set_xlabel('phase by PCA (h)', fontsize=12)
axx[3].set_ylabel('RORC', fontsize=12.)
axx[3].set_xticks([0, 4, 8, 12, 16, 20, 24])
axx[3].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
axx[3].grid()
axx[4].set_ylim(-2.5, 2.4)
axx[4].set_xlabel('phase by PCA (h)', fontsize=12)
axx[4].set_ylabel('PER3', fontsize=12.)
axx[4].set_xticks([0, 4, 8, 12, 16, 20, 24])
axx[4].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
axx[4].grid()
axx[5].set_ylim(-2.5, 2.4)
axx[5].set_xlabel('phase by PCA (h)', fontsize=12)
axx[5].set_ylabel('PER3', fontsize=12.)
axx[5].set_xticks([0, 4, 8, 12, 16, 20, 24])
axx[5].set_xticklabels([0, '', '', 12, '', '', 24], fontsize=10)
axx[5].grid()

