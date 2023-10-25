"""These tests verify that the dataset classes are functioning properly."""
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_frame_not_equal

import mokapot
from mokapot import PsmDataset, PsmSchema
from mokapot.proteins import Proteins


def test_psm_dataset_init(psm_df_6):
    """Test that a PsmDataset initializes correctly."""
    df, schema_kwargs = psm_df_6
    schema = PsmSchema(**schema_kwargs)
    dset = PsmDataset(df, schema)

    assert_frame_equal(dset.data.collect(), df)
    np.testing.assert_allclose(
        dset.features,
        df.select(schema.features).to_numpy(),
    )
    np.testing.assert_allclose(
        dset.targets,
        df["target"].to_numpy(),
    )

    assert dset.columns == df.columns
    assert dset.proteins is None
    assert dset.schema == schema
    assert isinstance(dset.rng, np.random.Generator)

    # Verify this is unitialized:
    assert dset.schema.score is None

    # Also test dictionary init.
    data = {"s": [1, 2], "p": ["a", "b"], "t": [True, False]}
    schema = PsmSchema("t", "s", "p", score="s", desc=False)
    dset = PsmDataset(data, schema)


def test_assign_confidence(psm_df_1000):
    """Test that assign_confidence() methods run."""
    data, fasta, schema_kwargs = psm_df_1000
    schema_kwargs["group"] = None
    dset = PsmDataset(
        data=data, schema=PsmSchema(**schema_kwargs), eval_fdr=0.05
    )

    # also try adding proteins:
    assert dset.proteins is None
    dset.assign_confidence()

    # Make sure it works when lower scores are better:
    data = data.with_columns(pl.col("score") * -1)
    dset = PsmDataset(
        data=data, schema=PsmSchema(**schema_kwargs), eval_fdr=0.05
    )

    assert dset.proteins is None
    dset.assign_confidence()

    # also try adding proteins:
    proteins = mokapot.read_fasta(
        fasta,
        missed_cleavages=0,
        rng=1,
    )
    dset.proteins = proteins
    dset.assign_confidence()

    # Verify that the group column works:
    schema_kwargs["group"] = "group"
    data = data.with_columns(pl.col("score") * -1)
    dset = PsmDataset(
        data=data, schema=PsmSchema(**schema_kwargs), eval_fdr=0.05
    )
    dset.assign_confidence()


def test_update_labels(psm_df_6):
    """Test that the _update_labels() methods are working."""
    df, schema_kwargs = psm_df_6
    schema = PsmSchema(**schema_kwargs)
    dset = PsmDataset(df, schema, eval_fdr=0.5)

    scores = np.array([6, 5, 3, 3, 2, 1])
    real_labs = np.array([1, 1, 0, -1, -1, -1])
    new_labs = dset.update_labels(scores)
    assert np.array_equal(real_labs, new_labs)


def test_best_feature(psm_df_6):
    """Test finding the best feature."""
    df, schema_kwargs = psm_df_6
    schema = PsmSchema(desc=False, **schema_kwargs)
    dset = PsmDataset(df, schema, eval_fdr=0.5)

    best_feat = dset.best_feature
    assert best_feat[0] == "feature_1"
    assert best_feat[1]


def test_proteins(psm_df_6, mock_proteins):
    """Test adding proteins."""
    df, schema_kwargs = psm_df_6
    schema = PsmSchema(**schema_kwargs)
    dset = PsmDataset(df, schema, eval_fdr=0.5)

    assert dset.proteins is None
    with pytest.raises(ValueError):
        dset.proteins = "blah"

    dset.proteins = Proteins(
        {"a": "A", "b": "B", "c": "B", "d": "A", "e": "B"},
        rng=1,
    )
    assert dset.proteins is not None


@pytest.mark.parametrize(
    ("idx", "train", "n_folds", "subset", "expected"),
    [
        (0, True, 4, None, 6),
        (2, True, 4, None, 6),
        (0, False, 4, None, 2),
        (2, False, 4, None, 2),
        (0, True, 2, None, 4),
        (0, True, 4, 4, 4),
    ],
)
def test_split(idx, train, n_folds, subset, expected):
    """Test splitting the data."""
    df = pl.DataFrame(
        {
            "target": [True, True, True, True, False, False, False, False],
            "spectrum": [1, 2, 3, 4, 1, 2, 3, 4],
            "peptide": ["a", "b", "a", "c", "d", "e", "f", "g"],
            "feature_1": [4, 3, 2, 1, 1, 0, 0, 0],
        }
    )

    schema = PsmSchema(
        target="target",
        spectrum="spectrum",
        peptide="peptide",
    )

    dset = PsmDataset(df, schema, subset=subset)
    assert len(dset) == 8

    fold = dset.fold(idx, train=train, folds=n_folds)
    assert len(fold) == expected

    next_fold = dset.fold(idx + 1, train=train, folds=n_folds)
    assert_frame_not_equal(fold.data, next_fold.data)

    fold_again = dset.fold(idx, train=train, folds=n_folds)
    assert_frame_equal(fold.data, fold_again.data)
