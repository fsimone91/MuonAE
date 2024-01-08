# - rebinning, cropping and normalization of histograms

import numpy as np
import json
from sklearn.preprocessing import normalize


### from dataframe to numpy
def get_hist_values(df):
    ### same as builtin "df['histo'].values" but convert strings to np arrays
    # input arguments:
    # - df: a dataframe containing histograms (assumed to be of a single type!)
    # note: this function works for both 1D and 2D histograms,
    #       the distinction is made based on whether or not 'Ybins' is present as a column in the dataframe
    #       update: 'Ybins' is also present for 1D histograms, but has value 1!
    # output:
    # a tuple containing the following elements:
    # - np array of shape (nhists,nbins) (for 1D) or (nhists,nybins,nxbins) (for 2D)
    # - np array of run numbers of length nhists
    # - np array of lumisection numbers of length nhists
    # warning: no check is done to assure that all histograms are of the same type!
    
    # check for corruption of data types (observed once after merging several csv files)
    if isinstance( df.at[0,'Xbins'], str ):
        raise Exception('ERROR in dataframe_utils.py / get_hist_values:'
                +' the "Xbins" entry in the dataframe is of type str, while a numpy int is expected;'
                +' check for file corruption.')
    # check dimension
    dim = 1
    if 'Ybins' in df.keys():
        if df.at[0,'Ybins']>1: dim=2
    # initializations
    nxbins = df.at[0,'Xbins']+2 # +2 for under- and overflow bins
    vals = np.zeros((len(df),nxbins))
    if dim==2: 
        nybins = df.at[0,'Ybins']+2
        vals = np.zeros((len(df),nybins,nxbins))
    ls = np.zeros(len(df))
    runs = np.zeros(len(df))
    # check data type of 'histo' field
    # (string in csv files, numpy array in parquet files)
    # case of string
    if isinstance( df.at[0,'histo'], str ):
        # loop over all entries
        for i in range(len(df)):
            try:
                # default encoding (with comma separation)
                jsonstr = json.loads(df.at[i,'histo'])
            except:
                # alternative encoding (with space separation)
                print(df.at[i,'histo'].replace(' ', ','))
                jsonstr = json.loads(df.at[i,'histo'].replace(' ', ','))
            hist = np.array(jsonstr)
            if dim==2: hist = hist.reshape((nybins,nxbins))
            vals[i,:] = hist
            ls[i] = int(df.at[i,'fromlumi'])
            runs[i] = int(df.at[i,'fromrun'])
    # case of numpy array
    if isinstance( df.at[0,'histo'], np.ndarray ):
        vals = np.vstack(df['histo'].values)
        ls = df['fromlumi'].values
        runs = df['fromrun'].values
    ls = ls.astype(int)
    runs = runs.astype(int)
    return (vals,runs,ls)

### rebinning of histograms
def rebin1D(hists, factor=None):
    ### perform rebinning on a set of histograms
    # input arguments:
    # - hists: a numpy array of shape (nhistograms,nbins) for 1D or (nhistograms,nybins,nxbins) for 2D
    # - factor: the rebinning factor (for 1D),
    if factor is None: return hists
    if(not hists.shape[1]%factor==0): 
        print('WARNING in hist_utils.py / rebinhists: no rebinning performed since no suitable reduction factor was given.'
             +' The rebinning factor ({}) is not a divisor of the number of bins ({})'.format(factor,hists.shape[1]))
        return hists
    (nhists,nbins) = hists.shape
    newnbins = int(nbins/factor)
    rebinned = np.zeros((nhists,newnbins))
    for i in range(newnbins):
        rebinned[:,i] = np.sum(hists[:,factor*i:factor*(i+1)],axis=1)
    return rebinned

def norm1D(hists):
    return normalize(hists, norm='l1', axis=1)

