# Changelog for mokapot  

## [0.8.0] - 2022-03-11

Thanks to @sambenfredj, @gessulat, @tkschmidt, and @MatthewThe for 
PR #44, which made these things happen!

### Added
- A new command line argument, `--max_workers`. This allows the
  cross-validation folds to be computed in parallel.
- The `PercolatorModel` class now has an `n_jobs` parameter, which 
  controls parallelization of the grid search.

### Changes
- Improved speed by using multiple jobs for grid search by default.
- Parallelization within `mokapot.brew()` now uses `joblib` 
  instead of `concurrent.futures`.

## [0.7.4] - 2021-09-03
### Changed
- Improved documentation and added warnings for `--subset_max_train`. Thanks
  @jspaezp!

## [0.7.3] - 2021-07-20
### Fixed
- Fixed bug where the `--keep_decoys` did not work with `--aggregate`. Also,
  added tests to cover this. Thanks @jspaezp!

## [0.7.2] - 2021-07-16  
### Added  
- `--keep_decoys` option to the command line interface. Thanks @jspaezp!
- Notes about setting a random seed to the Python API documentation. (Issue #30)
- Added more information about peptides that couldn't be mapped to proteins. (Issue #29) 

### Fixed  
- Loading a saved model with `mokapot.load_model()` would fail because of an
  update to Pandas that introduced a new exception. We've updated mokapot 
  accordingly.

### Changed  
- Updates to unit tests. Warnings are now treated as errors for system tests.

## [0.7.1] - 2021-03-22  
### Changed  
- Updated the build to align with
  [PEP517](https://www.python.org/dev/peps/pep-0517/)

## [0.7.0] - 2021-03-19  
### Added  
- Support for downstream peptide and protein quantitation with
  [FlashLFQ](https://github.com/smith-chem-wisc/FlashLFQ). This is accomplished
  through the `mokapot.to_flashlfq()` function or the `to_flashlfq()` method of
  `LinearConfidence` objects. Note that to support the FlashLFQ format, you'll
  need to specify additional columns in `read_pin()` or use a PepXML input file
  (`read_pepxml()`). 
- Added a top-level function for exporting confident PSMs, peptides, and
  proteins from one or more `LinearConfidence` objects as a tab-delimited file:
  `mokapot.to_txt()`.
- Added a top-level function for reading FASTA files for protein-level 
  confidence estimates: `mokapot.read_fasta()`.
- Tests accompanying the support for the features above.
- Added a "mokapot cookbook" to the documentation with helpful code snippets.

### Changed
- Corresponding with support for new formats, the `mokapot.read_pin()` function
  and the `LinearPsmDataset` constructor now have many new optional parameters.
  These specify the columns containing the metadata needed to write the added
  formats.
- Starting mokapot should be slightly faster for Python >= 3.8. We were able to
  eliminate the runtime call to setuptools, because of the recent addition of
  `importlib.metadata` to the standard library, saving a few hundred
  milliseconds.

## [0.6.2] - 2021-03-12  
### Added
- Now checks to verify there are no debugging print statements in the code
  base when linting.

### Fixed  
- Removed debugging print statements.

## [0.6.1] - 2021-03-11
### Fixed
- Parsing Percolator tab-delimited files with a "DefaultDirection" line.
- `Label` column is now converted to boolean during PIN file parsing. 
  Previously, problems occurred if the `Label` column was of dtype `object`.
- Parsing modifications from pepXML files were indexed incorrectly on the
  peptide string.

## [0.6.0] - 2021-03-03  
### Added  
- Support for parsing PSMs from PepXML input files.
- This changelog.

### Fixed  
- Parsing a FASTA file previously failed if an entry was not followed by a 
  sequence. Now, missing sequences are tolerated and a warning is given instead.  
- When the learned model was worse than the best feature and the lower scores
  were better for the best feature, assigning confidence would fail.  
- Easy access to grouped confidence estimates in the Python API were not working
  due to a typo.  
- Deprecation warnings from Pandas about the `regex` argument.  
- Sometimes peptides were removed as shared incorrectly when part of a protein
  group.  

### Changed  
- Refactored and added many new unit and system tests.
- New pull-requests must now improve or maintain test coverage.
- Improved error messages.
