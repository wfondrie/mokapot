"""Handle proteins for the picked protein FDR."""
import logging

LOGGER = logging.getLogger(__name__)


class Proteins:
    """Store protein sequences.

    This class stores the mapping of peptides to proteins and the mapping of
    target proteins to their corresponding decoy proteins.

    We recommend creating Proteins objects using the
    :py:func:`mokapot.read_fasta()` function.

    Parameters
    ----------
    decoy_prefix : str
        The prefix used to indicate a decoy protein in the description
        lines of the FASTA file.
    peptide_map : Dict[str, str]
        A dictionary mapping peptide sequences to the proteins that
        may have generated them.
    protein_map : Dict[str, str]
        A dictionary mapping decoy proteins to the target proteins from
        which they were generated.
    shared_peptides : Dict[str]
        A dictionary mapping shared peptides to the proteins that may have
        generated them.
    has_decoys : bool
        Did the FASTA file have decoy proteins in it?

    Attributes
    ----------
    decoy_prefix : str
        The prefix used to indicate a decoy protein in the description
        lines of the FASTA file.
    peptide_map : Dict[str, str]
        A dictionary mapping peptide sequences to the proteins that
        may have generated them.
    protein_map : Dict[str, str]
        A dictionary mapping decoy proteins to the target proteins from
        which they were generated.
    shared_peptides : Dict[str]
        A dictionary mapping shared peptides to the proteins that may have
        generated them.
    has_decoys : bool
        Did the FASTA file have decoy proteins in it?

    """

    def __init__(
        self,
        decoy_prefix,
        peptide_map,
        protein_map,
        shared_peptides,
        has_decoys,
    ):
        """Initialize a Proteins object"""
        self._decoy_prefix = decoy_prefix
        self._peptide_map = peptide_map
        self._shared_peptides = shared_peptides
        self._protein_map = protein_map
        self._has_decoys = has_decoys

    @property
    def decoy_prefix(self):
        return self._decoy_prefix

    @property
    def peptide_map(self):
        return self._peptide_map

    @property
    def protein_map(self):
        return self._protein_map

    @property
    def shared_peptides(self):
        return self._shared_peptides

    @property
    def has_decoys(self):
        return self._has_decoys
