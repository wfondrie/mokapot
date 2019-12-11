# mokapot
A flexible Python implementation of the [Percolator](http://percolator.ms)
algorithm.

The mokapot package is under active development. More documentation will be added soon!

## Try it:  
All you need for input is results from a search engine in Percolator INput
format (pin). With this, you can perform a similar analysis that you would with
Percolator from within the Python interpreter:

```Python
>>> import mokapot as mp
>>> input_psms = mp.read_pin("psms.pin")
>>> model = mp.MokapotSVM()
>>> psms, peptides = model.percolate()
```

## New features in mokapot compared to Percolator:  

- Perform cross-validation by file, yielding per-file PSM and peptide
  statistics.  
- Analyze cross-linking mass spectrometry data using an appropriate method to
  estimate error rates.  
- Easily perform analyses directly with Python, including in Jupyter notebooks.  
- Change the number of cross-validation splits.

## What mokapot does not have currently:  
- Protein inference with the FIDO algorithm.  
- Estimation of posterior error probabilities (PEPs).  
