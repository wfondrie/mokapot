# Changelog for mokapot  

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
