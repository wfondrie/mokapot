import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

from . import __version__
from .confidence import LinearConfidence
from .dataset import OnDiskPsmDataset
from .utils import get_unique_peptides_from_psms, merge_sort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_psms", type=str, required=True)
    parser.add_argument("--decoys_psms", type=str, required=True)
    parser.add_argument("--test_fdr", type=float, default=0.01)
    parser.add_argument(
        "--keep_decoys",
        action="store_true",
        default=False,
    )
    parser.add_argument("--dest_dir", type=str)
    parser.add_argument(
        "--verbosity", type=int, choices=[0, 1, 2, 3], default=2
    )

    args = parser.parse_args()

    verbosity_dict = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    if verbosity_dict[args.verbosity] != logging.DEBUG:
        warnings.filterwarnings("ignore")
    logging.basicConfig(
        format=("[{levelname}] {message}"),
        style="{",
        level=verbosity_dict[args.verbosity],
    )

    logging.info("mokapot version %s", str(__version__))
    logging.info("Command issued:")
    logging.info("%s", " ".join(sys.argv))
    logging.info("")

    psms = OnDiskPsmDataset(
        target_column="Label",
        peptide_column="peptide",
        filename=None,
        columns=None,
        spectrum_columns=None,
        protein_column=None,
        group_column=None,
        feature_columns=None,
        metadata_columns=None,
        filename_column=None,
        scan_column=None,
        specId_column=None,
        calcmass_column=None,
        expmass_column=None,
        rt_column=None,
        charge_column=None,
        spectra_dataframe=None,
    )

    iterable = merge_sort(
        paths=[args.decoys_psms, args.targets_psms],
        target_column=psms.target_column,
        col_score="score",
    )
    sep = "\t"
    metadata_columns = ["PSMId", "Label", "peptide", "score", "proteinIds"]
    output_columns = [
        "PSMId",
        "peptide",
        "score",
        "q-value",
        "posterior_error_prob",
        "proteinIds",
    ]
    peptides_path = "peptides.csv"
    with open(peptides_path, "w") as f_peptide:
        f_peptide.write(f"{sep.join(metadata_columns)}\n")
    unique_peptides = get_unique_peptides_from_psms(
        iterable=iterable,
        peptide_col_index=2,
        out_peptides=peptides_path,
        sep=sep,
    )
    logging.info("\t- Found %i unique peptides.", unique_peptides)

    out_targets, out_decoys = [
        os.path.split(in_path)[-1].rsplit(".", 1)[0] + ".peptides"
        for in_path in [args.targets_psms, args.decoys_psms]
    ]
    if args.dest_dir is not None:
        Path(args.dest_dir).mkdir(exist_ok=True)
        out_targets, out_decoys = [
            os.path.join(args.dest_dir, out_path)
            for out_path in [out_targets, out_decoys]
        ]
    with open(out_targets, "w") as fp:
        fp.write(f"{sep.join(output_columns)}\n")
    if args.keep_decoys:
        with open(out_decoys, "w") as fp:
            fp.write(f"{sep.join(output_columns)}\n")

    LinearConfidence(
        psms=psms,
        levels=["peptide"],
        level_paths=[peptides_path],
        out_paths=[[out_targets, out_decoys]],
        decoys=args.keep_decoys,
        eval_fdr=args.test_fdr,
        sep=sep,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
    except ValueError as e:
        logging.error(f"[Error] {e}")
        sys.exit(250)  # input failure
