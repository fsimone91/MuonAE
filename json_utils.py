#credits: https://github.com/LukaLambrecht/ML4DQMDC-PixelAE/blob/master/utils/json_utils.py

import os
import sys
import json
import numpy as np

#thisdir = os.path.abspath(os.path.dirname(__file__))
#jsondir = os.path.join(thisdir,'/jsons')
jsondir = './jsons'


def loadjson( jsonfile ):
    ### load the content of a json file into a python object
    # input arguments:
    # - jsonfile: the name (or full path if needed) to the json file to be read
    # output:
    # - an dict object as specified in the note below
    # note: the json file is supposed to contain an object like this example:
    #       { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }
    #       although no explicit checking is done in this function, 
    #       objects that don't have this structure will probably lead to errors further in the code
    if not os.path.exists(jsonfile):
        raise Exception('ERROR in json_utils.py / loadjson: requested json file {} does not seem to exist...'.format(jsonfile))
    with open(jsonfile) as f: jsondict = json.load(f)
    return jsondict

### checking if given run/lumi values are in a given json object
def injson_single( run, lumi, jsondict ):
    ### helper function for injson, only for internal use
    # input arguments:
    # - run and lumi are integers
    # - jsondict is an object loaded from a json file
    # output:
    # - boolean whether the run/lumi combination is in the json dict
    run = str(run)
    if not run in jsondict: return False
    lumiranges = jsondict[run]
    for lumirange in lumiranges:
        if( len(lumirange)==1 and lumirange[0]<0 ):
            return True
        if( lumi>=lumirange[0] and lumi<=lumirange[1] ): 
            return True
    return False

def injson( run, lumi, jsonfile=None, jsondict=None ):
    ### find if a run and lumi combination is in a given json file
    # input arguments:
    # - run and lumi: integers or (equally long) arrays of integers
    # - jsonfile: a path to a json file
    # - jsondict: a dict loaded from a json file
    #   note: either jsonfile or jsondict must not be None!
    # output: 
    # boolean or array of booleans (depending on run and lumi)
    
    # check the json object to use
    if( jsonfile is None and jsondict is None ):
        raise Exception('ERROR in json_utils.py / injson: both arguments jsonfile and jsondict are None. Specify one of both!')
    if( jsonfile is not None and jsondict is not None ):
        raise Exception('ERROR in json_utils.py / injson: both arguments jsonfile and jsondict are given, which leads to ambiguities. Omit one of both!')
    if jsondict is None:
        jsondict = loadjson( jsonfile )
        
    # check if single or multiple run/lumi combinations need to be assessed    
    if not hasattr(run,'__len__') and not isinstance(run,str):
        run = [run]; lumi = [lumi]
    res = np.zeros(len(run),dtype=np.int8)
    
    # check for all run/lumi combinations if they are in the json object
    for i,(r,l) in enumerate(zip(run,lumi)):
        if injson_single( r, l, jsondict ): res[i]=1
    res = res.astype(np.bool)
    if len(res)==1: res = res[0]
    return res

def isgolden(run, lumi):
    ### find if a run and lumi combination is in the golden json file
    # input arguments:
    # - run and lumi: either integers or (equally long) arrays of integers
    
    jsonloc2017 = os.path.join( jsondir, 'json_GOLDEN_2017.txt' ) 
    # ultralegacy reprocessing; from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt
    jsonloc2018 = os.path.join( jsondir, 'json_GOLDEN_2018.txt' )
    # legacy reprocessing; from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt
    jsonloc2022 = os.path.join( jsondir, 'json_GOLDEN_2022.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json
    jsonloc2023 = os.path.join( jsondir, 'json_GOLDEN_2023.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions23/Cert_Collisions2023_366442_370790_Golden.json
    isinjson =   (injson(run,lumi,jsonfile=jsonloc2017)
                + injson(run,lumi,jsonfile=jsonloc2018)
                + injson(run,lumi,jsonfile=jsonloc2022)
                + injson(run,lumi,jsonfile=jsonloc2023))
    return isinjson

def isdcson(run, lumi):
    ### find if a run and lumi combination is in DCS-only json file
    # input arguments:
    # - run and lumi: either integers or (equally long) arrays of integers
    
    jsonloc2017 = os.path.join( jsondir, 'json_DCSONLY_2017.txt' )
    # from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/DCSOnly/json_DCSONLY.txt
    jsonloc2018 = os.path.join( jsondir, 'json_DCSONLY_2018.txt' )
    # from: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/DCSOnly/json_DCSONLY.txt
    jsonloc2022 = os.path.join( jsondir, 'json_DCSONLY_2022.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions22/DCSOnly_JSONS/Cert_Collisions2022_355100_362760_eraBCDEFG_13p6TeV_DCSOnly_TkPx.json
    jsonloc2023 = os.path.join( jsondir, 'json_DCSONLY_2023.txt' )
    # from: /eos/user/c/cmsdqm/www/CAF/certification/Collisions23/DCSOnly_JSONS/Collisions23_13p6TeV_eraBCD_366403_370790_DCSOnly_TkPx.json
    isinjson =   (injson(run,lumi,jsonfile=jsonloc2017) 
                + injson(run,lumi,jsonfile=jsonloc2018)
                + injson(run,lumi,jsonfile=jsonloc2022)
                + injson(run,lumi,jsonfile=jsonloc2023))
    return isinjson

def select_golden(df):
    ### keep only golden lumisections in df
    dfres = df[ isgolden(df['fromrun'].values,df['fromlumi'].values) ]
    dfres.reset_index(drop=True,inplace=True)
    return dfres

def select_dcson(df):
    ### keep only lumisections in df that have DCS-bit on
    dfres = df[ isdcson(df['fromrun'].values,df['fromlumi'].values) ]
    dfres.reset_index(drop=True,inplace=True)
    return dfres
