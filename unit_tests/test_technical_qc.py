"""Tests for the default technical-QC panel script (`technical_qc_analysis.py`)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import technical_qc_analysis as tqc


def _frames(rows):
    df = pd.DataFrame(rows)
    sample_info = (df[['run', 'engine', 'condition2']]
                   .drop_duplicates().reset_index(drop=True))
    return df, sample_info


def test_disambiguate_runs_leaves_distinct_names_alone():
    df, si = _frames([
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'A'},
        {'run': 'r2', 'engine': 'peaks', 'condition2': 'B'},
    ])
    out_df, out_si = tqc._disambiguate_runs(df, si)
    assert list(out_df['run']) == ['r1', 'r2']
    assert list(out_si['run']) == ['r1', 'r2']


def test_disambiguate_runs_tags_shared_run_names():
    df, si = _frames([
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'PEAKS arm'},
        {'run': 'r1', 'engine': 'diann', 'condition2': 'DIA-NN arm'},
        {'run': 'r2', 'engine': 'peaks', 'condition2': 'other'},
    ])
    out_df, out_si = tqc._disambiguate_runs(df, si)
    assert list(out_df['run']) == ['r1 [PEAKS arm]', 'r1 [DIA-NN arm]', 'r2']
    sets = out_si.groupby('condition2')['run'].apply(set)
    assert not (sets['PEAKS arm'] & sets['DIA-NN arm'])


def test_disambiguate_runs_raises_when_engine_cannot_separate():
    df, si = _frames([
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'A'},
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'B'},
    ])
    with pytest.raises(ValueError, match='more than one condition'):
        tqc._disambiguate_runs(df, si)


_PLATE_RUNS = ['e_Q1_A1', 'e_Q1_B2', 'e_Q2_A1', 'e_Q2_C3', 'e_Q3_D4']


@pytest.fixture
def one_search(monkeypatch):
    monkeypatch.setattr(tqc, 'CONDITIONS', {'search': 'features.csv'})
    monkeypatch.setattr(tqc, '_runs_of', lambda path: list(_PLATE_RUNS))
    return {'search': 'features.csv'}


def test_split_by_plate_field(one_search):
    out = tqc.split_conditions(one_search, 'plate_quadrant')
    assert sorted(out) == ['Q1', 'Q2', 'Q3']
    assert out['Q1'] == ('features.csv', ['e_Q1_A1', 'e_Q1_B2'])
    assert out['Q3'] == ('features.csv', ['e_Q3_D4'])


def test_split_by_regex_capture_group(one_search):
    out = tqc.split_conditions(one_search, r'_(Q[1-3])_')
    assert sorted(out) == ['Q1', 'Q2', 'Q3']
    assert tqc.condition_path('search') == 'features.csv'


def test_split_by_regex_requires_one_capture_group(one_search):
    with pytest.raises(ValueError, match='one capture group'):
        tqc.split_conditions(one_search, r'_(Q)([1-3])_')


def test_split_by_label_map_can_merge_groups(one_search):
    out = tqc.split_conditions(one_search, {'top': r'_Q[12]_', 'bottom': r'_Q3_'})
    assert out['top'][1] == ['e_Q1_A1', 'e_Q1_B2', 'e_Q2_A1', 'e_Q2_C3']
    assert out['bottom'][1] == ['e_Q3_D4']


def test_split_by_label_map_rejects_a_run_in_two_conditions(one_search):
    with pytest.raises(ValueError, match='cannot be in two conditions'):
        tqc.split_conditions(one_search, {'a': r'_Q1_', 'b': r'_A1$'})


def test_split_raises_when_nothing_matches(one_search):
    with pytest.raises(ValueError, match='matched none'):
        tqc.split_conditions(one_search, r'_(Z[1-9])_')


def test_split_prefixes_labels_when_several_folders(monkeypatch):
    monkeypatch.setattr(tqc, '_runs_of', lambda path: ['e_Q1_A1', 'e_Q2_A1'])
    out = tqc.split_conditions({'fast': 'a.csv', 'slow': 'b.csv'},
                               'plate_quadrant')
    assert sorted(out) == ['fast_Q1', 'fast_Q2', 'slow_Q1', 'slow_Q2']
    assert out['slow_Q1'] == ('b.csv', ['e_Q1_A1'])


def test_condition_accessors_accept_both_forms(monkeypatch):
    monkeypatch.setattr(tqc, 'CONDITIONS',
                        {'plain': 'a.csv', 'split': ('b.csv', ['r1'])})
    assert tqc.condition_path('plain') == 'a.csv'
    assert tqc.condition_runs('plain') is None
    assert tqc.condition_path('split') == 'b.csv'
    assert tqc.condition_runs('split') == ['r1']


def test_build_experiments_restricts_a_split_condition(monkeypatch, tmp_path):
    monkeypatch.setattr(tqc, 'DATA_ROOT', str(tmp_path))
    monkeypatch.setattr(tqc, 'CONDITIONS',
                        {'Q1': ('features.csv', ['e_Q1_A1', 'e_Q1_B2'])})
    monkeypatch.setattr(tqc, 'COND_ORDER', ['Q1'])
    monkeypatch.setattr(tqc, '_runs_of', lambda path: list(_PLATE_RUNS))
    monkeypatch.setattr(tqc.core, 'detect_engine', lambda path: 'peaks')
    exps = tqc.build_experiments()
    assert [e['file_tags'] for e in exps] == ['e_Q1_A1', 'e_Q1_B2']
    assert {e['condition2'] for e in exps} == {'Q1'}


def test_add_plate_position_fills_the_pc_factor_columns():
    si = pd.DataFrame({'run': ['e_Q1_A1', 'e_Q2_H12', 'nameless']})
    out = tqc._add_plate_position(si)
    assert list(out['plate_quadrant']) == ['Q1', 'Q2', None]
    assert list(out['plate_row'])[:2] == ['A', 'H']
    assert list(out['plate_col'])[:2] == [1, 12]
    for _, col, _ in tqc.PC_FACTORS:
        if col.startswith('plate_'):
            assert col in out.columns


def test_factor_r2_ignores_runs_missing_the_factor():
    y = [0.0, 0.1, 5.0, 5.1, 0.05, 5.05]
    groups = ['A', 'A', None, None, 'A', None]
    assert pd.isna(tqc._factor_r2(y, groups, 'categorical'))
    both = tqc._factor_r2(y, ['A', 'A', 'B', 'B', 'A', 'B'], 'categorical')
    assert both > 0.99


def test_factor_r2_is_bias_corrected_for_level_count():
    rng = np.random.default_rng(0)
    many = [f'L{i // 2}' for i in range(16)]
    raw, adj = [], []
    for _ in range(300):
        y = rng.normal(size=16)
        raw.append(tqc._eta_squared(y, np.asarray(many)))
        adj.append(tqc._factor_r2(y, many, 'categorical'))
    assert np.mean(raw) == pytest.approx(7 / 15, abs=0.05)
    assert np.mean(adj) < 0.15

    real = ['A'] * 8 + ['B'] * 8
    strong = np.where(np.array(real) == 'A', 0.0, 10.0) + rng.normal(size=16)
    assert tqc._factor_r2(strong, real, 'categorical') > 0.9


def test_factor_r2_of_pure_noise_is_near_zero_for_both_kinds():
    rng = np.random.default_rng(1)
    y = rng.normal(size=40)
    cat = tqc._factor_r2(y, [f'g{i % 4}' for i in range(40)], 'categorical')
    con = tqc._factor_r2(y, list(range(40)), 'continuous')
    assert cat < 0.15 and con < 0.15


def test_n_corrected_cv_rescales_toward_the_target_n():
    assert tqc.n_corrected_cv(10.0, 4, n_to=4) == pytest.approx(10.0)
    up = tqc.n_corrected_cv(10.0, 4, n_to=8)
    assert up > 10.0 and up == pytest.approx(10.721, abs=1e-3)
    assert tqc.n_corrected_cv(up, 8, n_to=4) == pytest.approx(10.0)


def test_sd_median_bias_matches_the_sampling_distribution():
    rng = np.random.default_rng(20260908)
    for n in (2, 4, 8, 12, 24):
        s = rng.standard_normal((200_000, n)).std(axis=1, ddof=1)
        assert tqc.sd_median_bias(n) == pytest.approx(np.median(s), abs=4e-3)


def test_sd_median_bias_rises_toward_one_and_is_defined_past_ten():
    vals = [tqc.sd_median_bias(n) for n in range(2, 40)]
    assert all(b < a < 1.0 for b, a in zip(vals, vals[1:]))
    assert tqc.sd_median_bias(12) == pytest.approx(0.969583, abs=1e-5)
    assert tqc.sd_median_bias(96) == pytest.approx(1.0, abs=6e-3)


def test_sd_median_bias_is_undefined_for_a_single_value():
    assert np.isnan(tqc.sd_median_bias(1))
    assert np.isnan(tqc.sd_median_bias(0))


def test_raw_file_map_finds_a_raw_subfolder(monkeypatch, tmp_path):
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw' / 'run_A.wiff').write_text('x')
    (tmp_path / 'raw' / 'run_B.raw').write_text('x')
    monkeypatch.setattr(tqc, 'DATA_ROOT', str(tmp_path))
    found = tqc._raw_file_map()
    assert set(found) == {'run_A', 'run_B'}


def test_raw_file_map_prefers_the_data_root_over_a_subfolder(monkeypatch, tmp_path):
    (tmp_path / 'run_A.wiff').write_text('x')
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw' / 'run_A.wiff').write_text('x')
    monkeypatch.setattr(tqc, 'DATA_ROOT', str(tmp_path))
    assert tqc._raw_file_map()['run_A'] == str(tmp_path / 'run_A.wiff')


def test_acquisition_stamps_fall_back_to_mtime(monkeypatch, tmp_path):
    f = tmp_path / 'run_A.raw'
    f.write_text('x')
    monkeypatch.setattr(tqc, 'ACQ_TIME_SOURCE', 'auto')
    stamps, source = tqc._acquisition_stamps({'run_A': str(f)})
    assert source == 'mtime'
    assert stamps['run_A'] == pytest.approx(f.stat().st_mtime)


def test_copied_gap_flags_clustered_stamps():
    import numpy as np
    t = np.array([0, 763, 764, 770, 800, 836, 1219, 1255], float)
    assert tqc._copied_gap(t) is not None


def test_copied_gap_passes_a_real_series():
    import numpy as np
    t = np.arange(8) * 36 * 60.0
    assert tqc._copied_gap(t) is None
    t[1] = 5.0
    assert tqc._copied_gap(t) is None


def test_copied_gap_needs_two_stamps():
    assert tqc._copied_gap([1.0]) is None

def test_raw_file_map_restricts_to_the_runs_it_is_given(monkeypatch, tmp_path):
    (tmp_path / 'raw').mkdir()
    for stem in ('run_A', 'run_B', 'run_unsearched'):
        (tmp_path / 'raw' / f'{stem}.wiff').write_text('x')
    monkeypatch.setattr(tqc, 'DATA_ROOT', str(tmp_path))

    assert set(tqc._raw_file_map()) == {'run_A', 'run_B', 'run_unsearched'}
    assert set(tqc._raw_file_map(['run_A', 'run_B'])) == {'run_A', 'run_B'}


def test_raw_file_map_ignores_requested_runs_with_no_raw_file(monkeypatch, tmp_path):
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw' / 'run_A.wiff').write_text('x')
    monkeypatch.setattr(tqc, 'DATA_ROOT', str(tmp_path))
    assert set(tqc._raw_file_map(['run_A', 'run_missing'])) == {'run_A'}


def _cv_screen_frames():
    rows = []
    for run, a, b in (('r1', 100.0, 50.0), ('r2', 110.0, 55.0), ('r3', 105.0, 52.0)):
        rows.append({'run': run, 'engine': 'peaks', 'condition2': 'A',
                     'precursor_id': 'p1', 'peptide_id': 'pep1',
                     'protein_group': 'PG1', 'precursor_intensity': a,
                     'peptide_intensity': a, 'pg_intensity': a})
        rows.append({'run': run, 'engine': 'peaks', 'condition2': 'A',
                     'precursor_id': 'p2', 'peptide_id': 'pep2',
                     'protein_group': 'PG1', 'precursor_intensity': np.nan,
                     'peptide_intensity': b, 'pg_intensity': a})
    return _frames(rows)


@pytest.fixture()
def cv_screen_summary(monkeypatch, tmp_path):
    df, si = _cv_screen_frames()
    monkeypatch.setattr(tqc, 'OUT_DIR', str(tmp_path))
    monkeypatch.setattr(tqc, 'SHOW_FIGURES', False)
    monkeypatch.setattr(tqc, 'COND_ORDER', ['A'])
    monkeypatch.setattr(tqc, 'COLOR', {'A': '#FA7A61'})
    return tqc.cv_screen(df, si), tmp_path


def test_cv_screen_counts_peptides_within_the_quantified_precursors(cv_screen_summary):
    summary, _ = cv_screen_summary
    row = summary.iloc[0]
    assert row['mean_precursors_per_run'] == 1.0
    assert row['mean_peptides_per_run'] == 1.0
    assert row['mean_protein_groups_per_run'] == 1.0


def test_cv_screen_light_bar_is_the_union_not_the_per_run_mean(
        monkeypatch, tmp_path):
    df, si = _cv_screen_frames()
    extra = df[(df['run'] == 'r1') & (df['precursor_id'] == 'p1')].copy()
    extra[['precursor_id', 'peptide_id', 'protein_group']] = ['p3', 'pep3', 'PG3']
    df = pd.concat([df, extra], ignore_index=True)
    monkeypatch.setattr(tqc, 'OUT_DIR', str(tmp_path))
    monkeypatch.setattr(tqc, 'SHOW_FIGURES', False)
    monkeypatch.setattr(tqc, 'COND_ORDER', ['A'])
    monkeypatch.setattr(tqc, 'COLOR', {'A': '#FA7A61'})
    row = tqc.cv_screen(df, si).iloc[0]
    for lv in ('precursors', 'peptides', 'protein_groups'):
        assert row[f'union_{lv}'] == 2
        assert row[f'mean_{lv}_per_run'] == pytest.approx(4 / 3)
    src = pd.read_csv(tmp_path / f'{tqc.PANEL_NAME}_sourcedata.csv')
    light = src[src['series'] == 'bar height (light)'].iloc[0]
    assert light['run'] == 'UNION'
    assert light['precursors'] == 2 and light['protein_groups'] == 2


def test_cv_screen_summary_keeps_its_published_column_names(cv_screen_summary):
    summary, _ = cv_screen_summary
    for col in ('mean_precursors_per_run', 'precursors_cv20',
                'median_precursor_cv_pct', 'median_precursor_cv_pct_n8',
                'frac_precursor_cv_below_20',
                'mean_protein_groups_per_run', 'protein_groups_cv20',
                'median_pg_cv_pct', 'median_pg_cv_pct_n8',
                'frac_pg_cv_below_20',
                'mean_peptides_per_run', 'peptides_cv20',
                'median_peptide_cv_pct', 'median_peptide_cv_pct_n8',
                'frac_peptide_cv_below_20'):
        assert col in summary.columns, col


def test_cv_screen_writes_the_named_panel_and_nothing_else(cv_screen_summary):
    _, out_dir = cv_screen_summary
    assert tqc.PANEL_NAME == 'ids_and_cv_by_level'
    for ext in ('pdf', 'png', '_sourcedata.csv'):
        sep = '' if ext.startswith('_') else '.'
        assert (out_dir / f'{tqc.PANEL_NAME}{sep}{ext}').exists()
    assert not list(out_dir.glob('cv_screen_*'))


def test_cv_screen_source_data_carries_every_level_not_just_the_drawn_one(
        cv_screen_summary):
    _, out_dir = cv_screen_summary
    src = pd.read_csv(out_dir / f'{tqc.PANEL_NAME}_sourcedata.csv')
    ecdf = src[src['panel'] == 'ecdf']
    assert set(ecdf['series']) == {'precursor', 'peptide', 'protein_group'}
    drawn = set(ecdf.loc[ecdf['drawn'], 'series'])
    assert drawn == {tqc.CV_ECDF_LEVEL.replace('protein', 'protein_group')}


def _one_bar_panel(fig_w, bar_top=100.0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.bar(0, bar_top, width=0.6)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, bar_top * 1.18)
    t = ax.text(0, bar_top * 0.99, '146,008\nCV<20%', ha='center', va='top',
                fontsize=8, color='white')
    return fig, ax, t


def test_inner_bar_label_is_rotated_when_wider_than_the_bar():
    fig, ax, t = _one_bar_panel(fig_w=0.8)
    tqc._fit_inner_bar_labels(fig, [(ax, [(t, 100.0)])])
    assert t.get_rotation() == 90
    assert '\n' not in t.get_text()


def test_inner_bar_label_is_left_alone_when_it_fits():
    fig, ax, t = _one_bar_panel(fig_w=12)
    tqc._fit_inner_bar_labels(fig, [(ax, [(t, 100.0)])])
    assert t.get_rotation() == 0
    assert t.get_text() == '146,008\nCV<20%'


def test_upright_label_taller_than_its_bar_moves_out_and_turns_black():
    fig, ax, t = _one_bar_panel(fig_w=0.8, bar_top=100.0)
    tqc._fit_inner_bar_labels(fig, [(ax, [(t, 1.0)])])
    assert t.get_rotation() == 90
    assert t.get_color() == 'black'
    assert t.get_position()[1] > 1.0
