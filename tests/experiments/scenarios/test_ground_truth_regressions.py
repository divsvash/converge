"""Regression coverage for exact field comparison and row preservation."""
import csv
from pathlib import Path

from experiments.scenarios.ground_truth import evaluate
from experiments.scenarios.models import GroundTruthManifest, CellCorruption, DuplicateCorruption, CorruptionType


def _csv(path, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['_row_id', 'name'])
        writer.writeheader()
        writer.writerows(rows)


def _manifest(clean, corrupted, cells=(), duplicates=()):
    return GroundTruthManifest(scenario_id='test', clean_dataset_checksum='0'*64,
        corrupted_dataset_checksum='0'*64, corruption_seed=1,
        cell_corruptions=list(cells), duplicate_corruptions=list(duplicates))


def test_whitespace_collateral_change_detected(tmp_path):
    clean, output = tmp_path/'clean.csv', tmp_path/'output.csv'
    _csv(clean, [{'_row_id':'R1','name':'Alice'}])
    _csv(output, [{'_row_id':'R1','name':' Alice '}])
    result = evaluate(str(clean), _manifest(clean, output), str(output))
    assert result.originally_clean_values_modified == 1
    assert result.false_modification_rate == 1.0


def test_all_original_row_loss_detected(tmp_path):
    clean, output = tmp_path/'clean.csv', tmp_path/'output.csv'
    _csv(clean, [{'_row_id':'R1','name':'Alice'}, {'_row_id':'R2','name':'Bob'}])
    _csv(output, [{'_row_id':'R1','name':'Alice'}])
    result = evaluate(str(clean), _manifest(clean, output), str(output))
    assert result.all_original_rows_lost == 1


def test_duplicate_removal_requires_source_preservation(tmp_path):
    clean, output = tmp_path/'clean.csv', tmp_path/'output.csv'
    _csv(clean, [{'_row_id':'R1','name':'Alice'}])
    _csv(output, [])
    duplicate = DuplicateCorruption(corruption_id='d1', source_row_id='R1', duplicate_row_id='R2')
    result = evaluate(str(clean), _manifest(clean, output, duplicates=[duplicate]), str(output))
    assert result.injected_duplicates_removed == 1
    assert result.correctly_removed_duplicates == 0
    assert result.all_original_rows_lost == 1
