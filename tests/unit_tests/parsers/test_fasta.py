"""Test that we can parse a FASTA file correctly."""
import pytest

import mokapot
from mokapot import digest, make_decoys, read_fasta


@pytest.fixture
def protein():
    """A tryptic protein and its peptides with 2 missed cleavages."""
    prot = "AAAAAKBBBBBRPCCCCCKRDDDDKEEEEEE"
    peps = {
        "AAAAAK",
        "AAAAAKBBBBBR",
        "AAAAAKBBBBBRPCCCCCK",
        "BBBBBR",
        "BBBBBRPCCCCCK",
        "BBBBBRPCCCCCKR",
        "PCCCCCK",
        "PCCCCCKR",
        "PCCCCCKRDDDDK",
        "RDDDDK",
        "RDDDDKEEEEEE",
        "DDDDKEEEEEE",
        "EEEEEE",
    }
    return prot, peps


@pytest.fixture
def missing_fasta(tmp_path):
    """Create a fasta file with a missing entry."""
    out_file = tmp_path / "missing.fasta"
    with open(out_file, "w+") as fasta_ref:
        fasta_ref.write(
            ">sp|test_1|test_1\n"
            ">sp|test_2|test_2\n"
            "TKDIPIIFLSAVNIDKRFITKGYNSGGADY"
        )

    return out_file


@pytest.fixture
def target_fasta(tmp_path):
    """A simple target FASTA."""
    out_file = tmp_path / "target.fasta"
    with open(out_file, "w+") as fasta_ref:
        fasta_ref.write(
            ">wf|target1\n"
            "MABCDEFGHIJKLMNOPQRSTUVWXYZKAAAAABRAAABKAAB\n"
            ">wf|target2\n"
            "MZYXWVUTSRQPONMLKJIHGFEDCBAKAAAAABRABABKAAB\n"
            ">wf|target3\n"
            "A" + "".join(["AB"] * 24) + "AK\n"
            ">wf|target4\n"
            "MABCDEFGHIJK"
        )

    return out_file


@pytest.fixture
def decoy_fasta(tmp_path):
    """A simple decoy FASTA."""
    out_file = tmp_path / "decoy.fasta"
    with open(out_file, "w+") as fasta_ref:
        fasta_ref.write(
            ">decoy_wf|target1\n"
            "MAFGHDCBEIJKLPMQNORSUYTVXWZKAAAABARAABAKABA\n"
            ">decoy_wf|target2\n"
            "MZYSVUXWTRQMPOLNKJGIFHBEDCAKAAAABARABBAKABA\n"
            ">wf|target3\n"
            "A" + "".join(["BA"] * 24) + "AK\n"
            ">decoy_wf|target4\n"
            "MAFGHDCBEIJK"
        )

    return out_file


def test_fasta_with_missing(missing_fasta):
    """Test that a fasta file can be parsed with missing entries.

    See https://github.com/wfondrie/mokapot/issues/13
    """
    read_fasta(missing_fasta)


def test_target_fasta(target_fasta):
    """Test that a FASTA file with only targets works."""
    long_pep = "A" + "".join(["AB"] * 24) + "AK"
    short_pep = "AAABK"

    # First the default parameters
    prot = read_fasta(target_fasta)

    # Check the peptide_map
    # 0 missed cleavages
    assert "MABCDEFGHIJK" in prot.peptides
    # 1 missed cleavage
    assert "MABCDEFGHIJKLMNOPQR" in prot.peptides
    # 2 missed cleavages
    assert "MABCDEFGHIJKLMNOPQRSTUVWXYZK" in prot.peptides
    # too short
    assert short_pep not in prot.peptides
    # too long
    assert long_pep not in prot.peptides
    # Check shared peptides:
    assert 2 == prot.peptides["AAAAABR"][1]


def test_parameters(target_fasta):
    """Test that changing the parameters actually changes things."""
    long_pep = "A" + "".join(["AB"] * 24) + "AK"
    short_pep = "AAABK"

    prot = read_fasta(
        target_fasta,
        missed_cleavages=0,
        clip_nterm_methionine=True,
        min_length=3,
        max_length=60,
        decoy_prefix="rev_",
    )

    # Check the peptide_map
    # 0 missed cleavages
    assert "MABCDEFGHIJK" in prot.peptides
    assert "ABCDEFGHIJK" in prot.peptides
    # 1 missed cleavage
    assert "ABCDEFGHIJKLMNOPQR" not in prot.peptides
    # 2 missed cleavages
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZK" not in prot.peptides
    # too short
    assert short_pep in prot.peptides
    # too long
    assert long_pep in prot.peptides
    # grouped protein:
    groups = [v[0] for v in prot.peptides.values()]
    assert "wf|target1,wf|target4" in groups
    # Check shared peptides:
    assert 2 == prot.peptides["AAB"][1]


def test_decoy_fasta(target_fasta, decoy_fasta):
    """Test decoys can be provided and used."""
    # Try without targets:
    with pytest.raises(ValueError) as msg:
        read_fasta(decoy_fasta)
        assert str(msg).startswith("Only decoy sequences were found")

    # Now do with both:
    prot = read_fasta([target_fasta, decoy_fasta])

    # Check the peptide_map
    # A target sequence
    assert "MABCDEFGHIJK" in prot.peptides


def test_mc_digest(protein):
    """Test a tryptic digest with missed cleavages."""
    prot, peps = protein
    digested = digest(prot, missed_cleavages=2)
    assert digested == peps


def test_no_mc_digest(protein):
    """Test a tryptic digest without missed cleavages."""
    prot, peps = protein
    no_mc = []
    for pep in peps:
        seq = pep[:-1]
        if ("K" not in seq) and ("R" not in seq):
            no_mc.append(pep)

    peps = set(no_mc)
    digested = digest(prot, missed_cleavages=0)
    assert digested == peps


def test_short_digest(protein):
    """Test a tryptic digest allowing for shorter peptides."""
    prot, peps = protein
    peps.add("DDDDK")
    digested = digest(prot, missed_cleavages=2, min_length=2)
    assert digested == peps


def test_psup_digest(protein):
    """Test a tryptic digest with proline suppression."""
    prot, peps = protein
    no_p = []
    for pep in peps:
        if not pep.endswith("BR") and not pep.startswith("P"):
            no_p.append(pep)

    no_p += ["BBBBBRPCCCCCKRDDDDK", "AAAAAKBBBBBRPCCCCCKR"]
    peps = set(no_p)
    digested = digest(prot, enzyme_regex="[KR](?!P)", missed_cleavages=2)
    assert digested == peps


def test_make_decoys(target_fasta, tmp_path):
    """Test the make_decoys() function."""
    concat_file = str(tmp_path / "concat.fasta")
    make_decoys(target_fasta, concat_file)
    before = len(mokapot.parsers.fasta._parse_fasta_files(target_fasta))
    after = len(mokapot.parsers.fasta._parse_fasta_files(concat_file))
    assert 2 * before == after

    sep_file = str(tmp_path / "decoy.fasta")
    make_decoys(target_fasta, sep_file, concatenate=False)
    after = len(mokapot.parsers.fasta._parse_fasta_files(sep_file))
    assert before == after
