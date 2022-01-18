# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:27:03 2019

@author: Lidia
Code to interpret CeNGEN data
In order to run the code:
    -Select name of genes of interest for which would like to visualize expression (target_genes)
    -Change layout of plots to the number of genes of interest (layout in reads per cell plots and palette in tsne plot)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

dataset = pd.read_csv (r'C:\Users\Lidia\Desktop\Neuropeptide_network\Dataset\L4_neurons_TSVs\data.tsv', delimiter='\t',encoding='utf-8') 
genes_data = pd.read_csv (r'C:\Users\Lidia\Desktop\Neuropeptide_network\Dataset\L4_neurons_TSVs\genes.tsv', delimiter='\t',encoding='utf-8')
cells_data = pd.read_csv (r'C:\Users\Lidia\Desktop\Neuropeptide_network\Dataset\L4_neurons_TSVs\cells.tsv', delimiter='\t',encoding='utf-8', header=0, usecols=['Barcode', 'total_counts', 'Neuron.type', 'Tissue.type'])
#dataset = pd.read_csv(r'C:\Users\Worm Tracker\Desktop\CeNGEN-Data.csv')
#genes_data = pd.read_csv (r'C:\Users\Worm Tracker\Desktop\CeNGEN-GeneInfo.csv')
#cells_data = pd.read_csv (r'C:\Users\Worm Tracker\Desktop\CeNGEN-CellInfo.csv', header = 0, usecols = ['Sample', 'Barcode', 'total_counts', 'Neuron.type', 'Tissue.type'])
#%%Select interesting group of genes (in this case monoamine binding GPCRs)
dataset['Gene'] = genes_data['gene_short_name']
#target_genes = ('dop-1', 'dop-2', 'dop-3', 'ser-1', 'ser-4', 'ser-7', 'mod-1', 'ser-2', 'ser-3', 'ser-5', 'ser-6', 'dop-5', 'dop-4', 'dop-6', 'tyra-2', 'tyra-3', 'octr-1')
target_genes = ('mec-2', 'unc-1', 'sto-1', 'sto-2', 'sto-3', 'sto-4', 'sto-5', 'sto-6', 'unc-24','flr-1','acd-2','acd-3','acd-4','acd-5','delm-1','delm-2','asic-1','asic-2','mec-4','mec-10','deg-1','del-1','del-4','unc-105','egas-1','egas-3','egas-4','unc-8','degt-1','del-2','del-3','del-5','del-6','del-7','del-8','del-9','del-10')
target_data = dataset.loc[dataset['Gene'].isin(target_genes)]
list_neurons = cells_data['Barcode'].loc[cells_data['Tissue.type'] == 'Neuron'] #list with barcodes fora ll cells taht are neurons
#target_data = pd.DataFrame (np.random.randint(0,1000, size=(17,52429)), columns=list_neurons)
target_neurons = target_data.filter(items = list_neurons)
#%%Sum repeated neurons to plot
target_neurons = target_neurons.T #transpose matrix to have each gene as a column
row_names = cells_data['Neuron.type'].loc[cells_data['Tissue.type'] == 'Neuron'].values 
target_neurons.columns = dataset['Gene'].loc[dataset['Gene'].isin(target_genes)] #label columns with interesting genes (y axes)
target_neurons['Cells'] = row_names  #create column with cell names
target_neurons = target_neurons.set_index('Cells') # set cell names column as index (x axes)
target_neuronsp = target_neurons.groupby(by = target_neurons.index, axis=0 ).sum() # sum all reads per cell for the same cell name and gene (cell names were sampled several times)
#%%Normalize reads for total number of reads per neuron
neurons = cells_data.drop(cells_data[cells_data['Tissue.type'] != 'Neuron'].index) # select only neuron data from cell types data
neuron_reads = neurons.groupby(by = neurons['Neuron.type'], axis = 0).sum() # sum all the total cell counts for the same cell names that were sampled several times
#reads_neuron = cells_data['total_counts'].loc[cells_data['Tissue.type'] == 'Neuron']
target_neurons_norm = pd.DataFrame ((target_neuronsp.values)/neuron_reads.values*1e6,index=target_neuronsp.index, columns=target_neuronsp.columns) # normalise gene reads dividing them by total cell count for cell name
#%%Sum repeated cells non neurons
target_all = target_data.T #transpose matrix to have each gene as a column 
target_all.columns = dataset['Gene'].loc[dataset['Gene'].isin(target_genes)] #label columns with interesting genes (y axes)
target_all = target_all.drop (['Unnamed: 0', 'Gene'], axis=0)
rows = cells_data['Neuron.type'].values
target_all['Cells'] = rows  #create column with cell names
target_all = target_all.set_index('Cells') # set cell names column as index (x axes)
target_allp = target_all.groupby(by = target_all.index, axis=0 ).sum() # sum all reads per cell for the same cell name and gene (cell names were sampled several times)
target_allt = target_allp.replace(to_replace = (1,2), value = (0,0))
#%%Normalize reads for total number of reads per neuron
others_reads = cells_data.groupby(by = cells_data['Neuron.type'], axis = 0).sum() # sum all the total cell counts for the same cell names that were sampled several times
#reads_neuron = cells_data['total_counts'].loc[cells_data['Tissue.type'] == 'Neuron']
target_all_norm = pd.DataFrame ((target_allp.values)/others_reads.values*1e6,index=target_allp.index, columns=target_allp.columns) # normalise gene reads dividing them by total cell count for cell name
#%% Plot reads per neuron
plt.style.use('seaborn-white')
fig1 = target_neuronsp.plot(kind ='bar', subplots = True, sharex = False, layout= (28,1), figsize = (20, 200), logy = True)
plt.ylabel('Reads')
plt.show()
#%%Plot normalised reads per million for each interesting gene
fig2 = target_neurons_norm.plot(kind ='bar', subplots = True, sharex = False, layout= (28,1), figsize = (20, 200))
plt.ylabel('RPM')
plt.show()
#%% Plot thresholded normalised data
thresholded = target_neurons_norm.replace (to_replace = (1, 2), value = (0, 0))
fig1 = thresholded.plot(kind ='bar', subplots = True, sharex = False, layout= (37,1), figsize = (20, 200), logy = True)
plt.ylabel('Reads')
plt.show()
#%%Plot all cells and neurons normalised and thresholded
fig3 = target_allt.plot(kind ='bar', subplots = True, sharex = False, layout= (37,1), figsize = (20, 200), logy = True)
plt.ylabel('Reads')
plt.show()
#%%Perform t-SNE on data
tsnedata = target_allt.T
tSNE_normalised = TSNE(n_components=2, perplexity=30).fit_transform (tsnedata)
#tSNE_normalised.shape
tsnedata['tsne-2d-one'] = tSNE_normalised[:,0]
tsnedata['tsne-2d-two'] = tSNE_normalised[:,1]
tsnedata['target_genes'] = tsnedata.index
plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two", hue="target_genes", 
    palette=sns.color_palette("hls", 37),
    data=tsnedata,
    legend=False,
    alpha=1)
for i, point in enumerate(target_genes):
    x=tsnedata['tsne-2d-one']
    y=tsnedata['tsne-2d-two']
    ax.annotate(point, (x[i], y[i]))
plt.show()
#plt.figure(figsize=(16,10))
#sns.scatterplot(
#    x="tsne-2d-one", y="tsne-2d-two", hue="target_genes", 
#    palette=sns.color_palette("hls", 28),
#    data=tsnedata,
#    legend="full",
#    alpha=1
#)