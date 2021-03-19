The mokapot cookbook
####################

This page contains recipes---code snippets---to accomplish a variety of tasks
with mokapot. The idea behind these examples is to illustrate the use of
mokapot for common experimental designs provide starting points for conducting
your own customized analyses.

.. note::

   These recipes often make use of `Python list comprehensions
   <https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions>`_
   for brevity.

   .. code-block::

      # This list comprehension:
      y = [x for x in range(10)]

      # Is equivalent to:
      y = []
      for x in range(10):
          y.append(x)


.. contents::
   :depth: 2
   :local:


Python API workflows
--------------------

Analyze PSMs from a single file with mokapot
============================================

The simplest case is when we have a single file containing the PSMs identified
by a database search engine. If our file, :code:`psms.pin`, is in the
Percolator tab-delimited format, we can perform a mokapot analysis and save the
PSM and peptide confidence estimates as tab-delimited files with:

.. code-block:: python

   import mokapot

   # Read the PSMs from the PIN file:
   psms = mokapot.read_pin("psms.pin")

   # Conduct the mokapot analysis:
   results, models = mokapot.brew(psms)

   # Save the results to two tab-delimited files
   # "mokapot.psms.txt" and "mokapot.peptides.txt"
   result_files = results.to_txt()

   # Another way to save the results is:
   result_files = mokapot.to_txt(results)


We can do same the same if our input PSMs are in the PepXML format,
:code:`psms.pep.xml`, as well:

.. code-block:: python

   import mokapot

   # Read the PSMs from the PepXML file:
   psms = mokapot.read_pepxml("psms.pep.xml")

   # Conduct the mokapot analysis:
   results, models = mokapot.brew(psms)

   # Save the results to two tab-delimited files
   # "mokapot.psms.txt" and "mokapot.peptides.txt"
   result_files = results.to_txt()


Analyze PSMs from a single file using only the best feature
===========================================================

It is often good to determine if there is a significant benefit from using
mokapot's machine learning approach. One way we can do this is by comparing the
results from a full mokapot analysis against the results from ranking PSMs by
the best feature (often the primary database search engine score). If our file,
:code:`psms.pin`, is in the Percolator tab-delimited format, we can use the
best feature, then save the PSM and peptide confidence estimates as
tab-delimited files with:

.. code-block:: python

   import mokapot

   # Read the PSMs from the PIN file:
   psms = mokapot.read_pin("psms.pin")

   # Calculate confidence estimates using the best feature:
   results = psms.assign_confidence()

   # Save the results to two tab-delimited files
   # "mokapot.psms.txt" and "mokapot.peptides.txt"
   result_files = results.to_txt()


Analyze PSMs from a single file with protein-level results
==========================================================

We often want confidence estimates for proteins as well as PSMs and peptides.
In mokapot, we use the picked-protein approach to group proteins and assign
their confidence estimates. To enable these protein confidence estimates, we
need to provide the FASTA file and the digestion settings that we used for our
database search. If our file, :code:`psms.pin`, is in the Percolator
tab-delimited format and we obtained these PSMs using the :code:`human.fasta`
protein database with a full tryptic digest, we can perform our analysis with:

.. code-block::

   import mokapot

   # Read the PSMs from the PIN file:
   psms = mokapot.read_pin("psms.pin")

   # Provide the protein sequences:
   psms.add_proteins(
       "human.fasta",
       enzyme="[KR]",
       decoy_prefix="decoy_",
       missed_cleavages=0,
   )

   # Conduct the mokapot analysis:
   results, models = mokapot.brew(psms)

   # Save the results to three tab-delimited files
   # "mokapot.psms.txt", "mokapot.peptides.txt", and "mokapot.proteins.txt"
   result_files = results.to_txt()


Analyze PSMs from a single fractionated sample in multiple files
================================================================

Offline fractionation is typically performed to increase the detectable
proteome depth for a sample. Sometimes these types of analyses will yield
multiple files for the detected PSMs, each corresponding to a single mass
spectrometry run of the different biochemical fractions. If we have the PSMs
from three fractions, :code:`fraction_1.pin`, :code:`fraction_2.pin`, and
:code:`fraction_3.pin`, we can analyze them together in mokapot with:

.. code-block::

   import mokapot

   # Create a list with our file names:
   psm_files = ["fraction_1.pin", "fraction_2.pin", "fraction_3.pin"]

   # Read the PSMs from all of the files:
   psms = mokapot.read_pin(psm_files)

   # Conduct the mokapot analysis:
   results, models = mokapot.brew(psms)

   # Save the results to two tab-delimited files
   # "mokapot.psms.txt" and "mokapot.peptides.txt"
   result_files = results.to_txt()


Analyze PSMs from multiple experiments using a joint model
==========================================================

We often want to compare the detected peptides and proteins between multiple
biological samples or experiments. One way to conduct this type of analysis
with mokapot is to use a joint model, such that the model learned by mokapot is
consistent across experiments. If we have PSMs from three experiments,
:code:`exp_1.pin`, :code:`exp_2.pin`, :code:`exp_3.pin`, we can analyze them
using a joint model with:

.. code-block::

   import mokapot

   # Create a list with our file names:
   psm_files = ["exp_1.pin", "exp_2.pin", "exp_3.pin"]

   # Read the PSMs from each file separately:
   psm_list = [mokapot.read_pin(f) for f in psm_files]

   # Conduct the mokapot analysis:
   result_list, models = mokapot.brew(psm_list)

   # Save the results to two tab-delimited files for each experiment:
   # "exp_1.mokapot.psms.txt", "exp_1.mokapot.peptides.txt", ...
   labels = ["exp_1", "exp_2", "exp_3"]
   result_files = [r.to_txt(file_root=l) for l, r in zip(labels, result_list)]


Analyze PSMs from multiple experiments using independent models
===============================================================

Like above, we can alternatively analyze PSMs from multiple experiments each
with their own model. If we have PSMs from three experiments,
:code:`exp_1.pin`, :code:`exp_2.pin`, :code:`exp_3.pin`, we can analyze them
using independent models with:

.. code-block::

   import mokapot

   # Create a list with our file names:
   psm_files = ["exp_1.pin", "exp_2.pin", "exp_3.pin"]

   # Read the PSMs from each file separately:
   psm_list = [mokapot.read_pin(f) for f in psm_files]

   # Conduct the mokapot analyses separately:
   # This returns a nested list: [[exp_1_result, exp_1_models], ...]
   results_and_models = [mokapot.brew(p) for p in psm_list]

   # Unnest the nested list:
   result_list, models = list(zip(*results_and_models))

   # Save the results to two tab-delimited files for each experiment:
   # "exp_1.mokapot.psms.txt", "exp_1.mokapot.peptides.txt", ...
   labels = ["exp_1", "exp_2", "exp_3"]
   result_files = [r.to_txt(file_root=l) for l, r in zip(labels, result_list)]


Analyze PSMs from multiple experiments with multiple fractions
==============================================================

The previous cases of multiple experiments and multiple fractions are
frequently combined for deep proteomics datasets. Let's assume we have PSMs
from two experiments, each with two fractions: :code:`exp_1-fraction_1.pin`,
:code:`exp_1-fraction_2.pin`, :code:`exp_2-fraction_1.pin`,
:code:`exp_2-fraction_2.pin`. We can then analyze them using a joint model in
mokapot with:

.. code-block::

   import mokapot

   # Create a nested list with our file names:
   psm_file_groups = [
       ["exp_1-fraction_1.pin", "exp_1-fraction_2.pin"], # exp_1
       ["exp_2-fraction_1.pin", "exp_2-fraction_2.pin"], # exp_2
   ]

   # Read the PSMs from each experiment group separately:
   psm_list = [mokapot.read_pin(f) for f in psm_file_groups]

   # Conduct the mokapot analysis:
   result_list, models = mokapot.brew(psm_list)

   # Save the results to two tab-delimited files for each experiment:
   # "exp_1.mokapot.psms.txt", "exp_1.mokapot.peptides.txt", ...
   labels = ["exp_1", "exp_2"]
   result_files = [r.to_txt(file_root=l) for l, r in zip(labels, result_list)]


Save results for label-free quantitation with FlashLFQ
======================================================

`FlashLFQ <https://github.com/smith-chem-wisc/FlashLFQ>`_ is an open-source
tool for label-free quantitation of peptides and proteins. Unfortunately, input
files in the Percolator tab-delimited format typically do not contain enough
information to create an input file for FlashLFQ. Although these can be added
to the file and specified through the optional parameters of
:py:func:`mokapot.read_pin()`, we find it often easier to use a PepXML file,
which already contains this information. If we have PSMs from two experiments,
:code:`exp_1.pep.xml` and :code:`exp_2.pep.xml`, we can analyze them with
mokapot using a joint model and save the detected peptides in a format for
input to FlashLFQ with the following. Note that the protein groups reported by
FlashLFQ will be most accurate if protein-level confidence estimates have been
enabled.


.. code-block::

   import mokapot

   # Create a list with out file names:
   psm_files = ["exp_1.pep.xml", "exp_2.pep.xml"]

   # Read the PSMs from each experiment separately:
   psm_list = [mokapot.read_pepxml(f) for f in psm_files]

   # Read the proteins from a FASTA file and add them to each experiment:
   proteins = mokapot.read_fasta("human.fasta")
   [p.add_proteins(proteins) for p in psm_list]

   # Conduct the mokapot analysis:
   result_list, models = mokapot.brew(psm_list)

   # Save results as tab-delimited files:
   labels = ["exp_1", "exp_2"]
   result_files = [r.to_txt(file_root=l) for l, r in zip(labels, result_list)]

   # Create an input for FlashLFQ:
   flashlfq_file = mokapot.to_flashlfq(result_list)


The final command will create a file :code:`mokapot.flashlfq.txt` that we can
use to obtain quantitative results for the peptides and proteins using
FlashLFQ.


Python API Tips and Tricks
--------------------------

Turn on logging messages
========================

By default, mokapot will only print warnings and errors when using the Python
API. However, information about mokapot's progress can be enabled by adding
the following to the beginning of your script or notebook:

.. code-block::

   import logging

   logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


When the same FASTA file is used for multiple experiments
=========================================================

One way to add proteins from a FASTA file (:code:`human.fasta`) to a collection
of PSMs (:code:`psms.pin`") is:

.. code-block::

   import mokapot

   psms = mokapot.read_pin("psms.pin")
   psms.add_proteins("human.fasta")

Correspondingly, when we have PSMs from multiple experiments
(:code:`exp_1.pin`, :code:`exp_2.pin`, :code:`exp_3.pin`), we could do:

.. code-block::

   import mokapot

   psm_files = ["exp_1.pin", "exp_2.pin", "exp_3.pin"]
   psm_list = [mokapot.read_pin(f) for f in psm_files]

   # Add proteins to each independently:
   [p.add_proteins("human.fasta") for p in psm_list]

This will work; however, the protein and peptide sequences from
:code:`human.fasta` will be stored 3 separate times, despite containing the
same information.

Instead, it is much more memory efficient to use
:py:func:`mokapot.read_fasta()` once and add the resulting
:py:class:`~mokapot.proteins.Proteins` object to each of the experiments:

.. code-block::

   import mokapot

   psm_files = ["exp_1.pin", "exp_2.pin", "exp_3.pin"]
   psm_list = [mokapot.read_pin(f) for f in psm_files]

   # Read the FASTA file:
   proteins = mokapot.read_fasta("human.fasta")

   # Add the proteins to the experiments:
   [p.add_proteins(proteins) for p in psm_list]


Enzyme Regular Expressions
--------------------------

For maximum flexibility, mokapot uses `regular expressions
<https://www.rexegg.com/>`_ (regex for short) to define the patterns that
govern enzymatic cleavage in protein sequences. However, it can be
frustratingly difficult to write from scratch. In the table below, we list
regular expressions for some common enzymes used in proteomics experiments. If
the one you need is not listed, we recommend using `mokapot.digest()` to test a
new one on a sample sequence.

In mokapot, the end of the sequence matching the regex is used to define the
cleavage site.

=====================================   ======================
Enzyme                                  Regex
=====================================   ======================
Trypsin (without proline suppression)   :code:`"[KR]"`
Trypsin (with proline suppression)      :code:`"[KR](?!P)"`
Lys-C                                   :code:`"K(?!P)"`
Lys-N                                   :code:`".(?=K)"`
Arg-C                                   :code:`"R(?!P)"`
Asp-N                                   :code:`".(?=D)"`
CNBr                                    :code:`"M"`
Glu-C                                   :code:`"[DE](?!P)"`
PepsinA                                 :code:`"[FL](?!P)"`
Chymotrypsin                            :code:`"[FWYL](?!P)"`
=====================================   ======================

To indicate more than one enzyme, we can use regex alternations with
:code:`|`. For example, we could specify trypsin and chymotrypsin with:

.. code-block::

   "([KR](?!P)|[FWYL](?!P))"

In this case, we also have the option to simplify the regex to:

.. code-block::

   "[KRFWYL](?!P)"

Questions?
----------

Still have questions? Post them on our `discussion board
<https://github.com/wfondrie/mokapot/discussions>`_.

If you find a mistake---such as a typo or code that doesn't run---please let us
know by `filing an issue <https://github.com/wfondrie/mokapot/issues>`_. Also,
please consider :doc:`contributing <contributing>` if you know how to fix it.
