#!/usr/bin/python3
import sys
import os
sys.path.append('./')
# append nephosem
from nephosem import ConfigLoader, Vocab, TypeTokenMatrix
from nephosem import ItemFreqHandler, ColFreqHandler
from nephosem import compute_association, compute_distance
from sklearn_extra.cluster import KMedoids
import sklearn as sk
import pandas as pd
import pydra
import json
import argparse

@pydra.mark.task
def getVocab(settings, fnames = None):
  ifhan = ItemFreqHandler(settings)
  full = ifhan.build_item_freq(fnames=fnames)
  return full

@pydra.mark.task
def filterVocab(vocab, regex):
  return vocab[vocab.match('item', regex)]

@pydra.mask.task
@pydra.mark.annotate({"return": {
    "freqMTX": TypeTokenMatrix,
    "nfreq": Vocab,
    "cfreq" : Vocab}})
def getCollocates(settings, window_size, vocab):
    settings['left-span'] = window_size[0]
    settings['right-span'] = window_size[1]
    cfhan = ColFreqHandler(settings = settings, row_vocab = vocab)
    freqMTX = cfhan.build_col_freq()
    nfreq = Vocab(freqMTX.sum(axis=1))
    cfreq = Vocab(freqMTX.sum(axis=0))
    return freqMTX, nfreq, cfreq

@pydra.mask.task
def subsetMatrix(freqMTX, rows, vocab, length = None):
    length = len(vocab) if length is None else length
    return freqMTX.submatrix(row = rows, col = vocab[:length].get_item_list())

@pydra.mask.task
def weightMatrix(freqMTX, nfreq, cfreq, meas):
    return compute_association(
        freqMTX = freqMTX,
        nfreq = nfreq,
        cfreq = cfreq,
        meas = meas
    )

@pydra.mask.task
@pydra.mark.annotate({"return": {
    "freqMTX": TypeTokenMatrix,
    "meas": str}})
def reduceDimensions(freqMTX, meas, dimensionality):
    if dimensionality is None:
        return freqMTX, meas
    else:
        import umap
        mtx = umap.UMAP(
            metric = meas,
            n_components = dimensionality
        ).fit_transform(freqMTX.matrix)
        ttm = TypeTokenMatrix(
            mtx,
            freqMTX.row_items,
            ['dim_' + str(i+1) for i in range(dimensionality)])
        return ttm, 'euclidean'

@pydra.mask.task
def getDistance(freqMTX, meas):
    return compute_distance(freqMTX, metric = meas)

@pydra.mask.task
def getClustering(freqMTX, meas, k):
    clustering = KMedoids(n_clusters = k, metric = meas, method = 'pam')
    fit = clustering.fit(freqMTX.matrix)
    silhouettes = sk.metrics.silhouette_samples(
        freqMTX.row_items,
        fit.labels_,
        metric = meas)
    medoids = [freqMTX.row_items[x] for x in fit.medoid_indices_]
    medoid_assignment = [medoids[x] for x in fit.labels_]
    df = pd.DataFrame({
        'type_word' : freqMTX.row_items,
        'cluster_number' : fit.labels_,
        'silhouette' : silhouettes,
        'medoid_label' : medoid_assignment
    })
    return df

def createWorkflow(inputs, cache_dir, parameters):
    wf = pydra.Workflow(
            name = 'wf',
            input_spec = list(inputs.keys()),
            **inputs,
            cache_dir = cache_dir
        )
    wf.split(parameters)
    wf.add(getVocab(
        name = 'vocab',
        settings = wf.lzin.settings, fnames = wf.lzing.fnames
        ))
    wf.add(filterVocab(
        name = 'filter',
        vocab = wf.vocab.lzout.out,
        regex = wf.lzin.vocab_regex
        ))
    wf.add(getCollocates(
        name = 'colloc',
        settings = wf.lzin.settings,
        window_size = wf.lzin.window_size,
        vocab = wf.filter.lzout.out
    ))
    wf.add(subsetMatrix(
        name = 'submtx',
        freqMTX = wf.colloc.lzout.freqMTX,
        rows = wf.lzin.row_selection,
        vocab = wf.filter.lzout.out,
        length = wf.lzin.dimensionality
    ))
    wf.add(weightMatrix(
        name = 'weight',
        freqMTX = wf.submtx.lzout.out,
        nfreq = wf.colloc.lzout.nfreq,
        cfreq = wf.colloc.lzout.cfreq,
        meas = wf.lzin.assoc
    )),
    wf.add(reduceDimensions(
        name = 'dimred',
        freqMTX = wf.weight.lzout.out,
        meas = wf.lzin.dist,
        dimensionality = wf.lzin.dimred
    ))
    wf.add(getDistance(
        name = 'dist',
        freqMTX = wf.dimred.lzout.freqMTX,
        meas = wf.dimred.lzout.meas
    ))
    wf.add(getClustering(
        name = 'clus',
        freqMTX = wf.dist.lzout.out,
        meas = wf.dimred.lzout.meas,
        k = wf.lzin.k
    ))
    wf.set_output([('clustering', wf.clus.lzout.out)])
    with pydra.Submitter(plugin = 'cf') as sub:
        sub(runnable = wf)
    return wf

defaults = {
     'fnames' : None,
     'vocab_regex' : ['.*'],
     'window_size' : [(5, 5)],
     'dimensionality' : [5000],
     'assoc': ['ppmi'],
     'dist' : ['cosine'],
     'dimred' : [None],
     'k' : [8]
    }
def checkArgs(inputs):
    for key in defaults.keys():
        if not key in inputs.keys():
            print(f"Using default value of '{key}': '{defaults[key]}")
            inputs[key] = defaults[key]
        elif key != 'fnames' and type(inputs[key]) != list:
            Exception(f"Values should be lists but '{key}', please fix your Parameters file!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'parameters',
        type = str,
        help = "JSON-formated file with the parameters for modelling; values must be lists."
        )
    parser.add_argument(
        '-s',
        '--settings',
        type = str,
        default = os.path.join(os.getcwd(),'config.ini')
    )
    parser.add_argument(
        '-c',
        '--cachedir',
        type = str,
        default = os.path.join(os.getcwd(), 'cache')
    )
    parser.add_argument(
        '-o',
        '-outdir',
        type = str,
        default = os.getcwd()
    )
    args = parser.parse_args()
    
    for path in args:
        if not os.path.exists(path):
            Exception(f"{path} must exist!")
    
    with open(args.parameters, 'r') as f:
        inputs = json.load(f) # read from arguments
    checkArgs(inputs)
    # TODO Set required keys and send error if any is missing
    conf = ConfigLoader()
    settings = conf.update_config(args.settings)
    parameters = list(inputs.keys())
    inputs['settings'] = settings;
    wf = createWorkflow(inputs, args.cachedir, parameters)
    result = wf.result(return_inputs = True)
    params = pd.DataFrame([res[0] for res in result])
    params['cache_folder'] = wf.output_dir
    params_file = os.path.join(args.outdir, 'models.tsv')
    params.to_csv(params_file, sep = '\t', index = False)
    print(f"Parameter space stored as in {params_file}.")
    # outdir = args.outdir
    # TODO check how to store the data: both the inputs and the output!
    # Also, It would be possible to get mini workflows inside to get data: 
    # at token level, to store the token-specific data,
    # and the dim_red-cosine-clustering part to reuse between type level and token level
    # TODO how to restore the stored data?
    # file = wf.create_dotfile(type = 'detailed', name = 'wf-detailed', output_dir = './')
    # graph available from my fork only, because of the splitting

