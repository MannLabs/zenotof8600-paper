"""Tests for the SCIEX OS MRM-HR mass-table export."""
import numpy as np
import pandas as pd
import pytest

import zenotof8600_analytics as core
from zenotof8600_analytics.targeted_method import (CE_RULES,
                                                  SCIEX_MRMHR_LAYOUTS,
                                                  precursor_q1_mz,
                                                  sciex_mrmhr_table,
                                                  write_sciex_mrmhr_table)

SEQ = 'NEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA'


@pytest.fixture
def targets():
    return [{'compound_id': 'a', 'precursor_mz': 1093.9339, 'rt_min': 5.44},
            {'compound_id': 'b', 'precursor_mz': 1073.9423, 'rt_min': 5.34}]


def test_layout_headers_are_literal():
    assert SCIEX_MRMHR_LAYOUTS['cid_scheduled'] == (
        'Compound ID', 'Group name', 'Precursor Ion (Da)',
        'TOF Start Mass (Da)', 'TOF Stop Mass (Da)',
        'Accumulation Time (sec)', 'QJet DP (V)', 'Collision energy (V)',
        'CE Spread (V)', 'Retention time (min)',
        'Retention time tolerance (sec)')
    ead = SCIEX_MRMHR_LAYOUTS['ead']
    assert ead[4:7] == ('TOF Stop Mass (Da)', 'Reaction time (ms)',
                        'Accumulation Time (sec)')


def test_reaction_time_is_written_empty_not_dropped(targets):
    t = sciex_mrmhr_table(targets, layout='ead', collision_energy=12.0)
    assert 'Reaction time (ms)' in t.columns
    assert (t['Reaction time (ms)'] == '').all()
    assert t['Accumulation Time (sec)'].tolist() == ['0.0500', '0.0500']


def test_formatting_is_literal(targets):
    t = sciex_mrmhr_table(targets, layout='cid', collision_energy=49.5)
    assert t['Precursor Ion (Da)'].tolist() == ['1093.93390', '1073.94230']
    assert t['Collision energy (V)'].tolist() == ['49.5', '49.5']
    assert t['QJet DP (V)'].tolist() == ['20', '20']
    assert t['CE Spread (V)'].tolist() == ['0', '0']


def test_collision_energy_rule_and_callable(targets):
    by_call = sciex_mrmhr_table(targets, collision_energy=lambda mz: 0.048 * mz - 2)
    by_name = sciex_mrmhr_table(targets, collision_energy='linear_0048')
    assert by_call['Collision energy (V)'].tolist() == \
        by_name['Collision energy (V)'].tolist() == ['50.5', '49.5']
    assert CE_RULES['linear_0048'](1000.0) == pytest.approx(46.0)


def test_collision_energy_has_no_default(targets):
    with pytest.raises(ValueError, match='no default'):
        sciex_mrmhr_table(targets)


def test_per_row_overrides_beat_keywords(targets):
    targets[0]['collision_energy'] = 99.0
    targets[0]['group'] = 'other'
    t = sciex_mrmhr_table(targets, collision_energy=10.0, group='Group 1')
    assert t['Collision energy (V)'].tolist() == ['99.0', '10.0']
    assert t['Group name'].tolist() == ['other', 'Group 1']


def test_scheduled_layout_requires_rt(targets):
    for t in targets:
        t.pop('rt_min')
    with pytest.raises(ValueError, match='rt_min'):
        sciex_mrmhr_table(targets, layout='cid_scheduled', collision_energy=12.0)


def test_duplicate_compound_id_rejected(targets):
    targets[1]['compound_id'] = targets[0]['compound_id']
    with pytest.raises(ValueError, match='unique'):
        sciex_mrmhr_table(targets, collision_energy=12.0)


def test_unknown_layout_lists_the_known_ones(targets):
    with pytest.raises(ValueError, match='cid_scheduled'):
        sciex_mrmhr_table(targets, layout='nope', collision_energy=12.0)


def test_write_roundtrip_is_tab_separated(tmp_path, targets):
    path = tmp_path / 'method.txt'
    write_sciex_mrmhr_table(targets, path, layout='cid_scheduled',
                            collision_energy='linear_0048')
    lines = path.read_text(encoding='utf8').splitlines()
    assert lines[0].split('\t') == list(SCIEX_MRMHR_LAYOUTS['cid_scheduled'])
    assert len(lines) == 3
    assert lines[1].split('\t')[0] == 'a'
    assert lines[1].split('\t')[-2:] == ['5.44', '30']


def test_write_without_header(tmp_path, targets):
    path = tmp_path / 'method.txt'
    write_sciex_mrmhr_table(targets, path, collision_energy=12.0, header=False)
    assert len(path.read_text(encoding='utf8').splitlines()) == 2


def _frame(mods, sites):
    return pd.DataFrame({'sequence': [SEQ], 'mods': [mods],
                         'mod_sites': [sites], 'charge': [4]})


def test_q1_monoisotope_is_the_default():
    df = _frame('Phospho@S', '27')
    out = precursor_q1_mz(df)
    assert out['q1_isotope_index'].tolist() == [0]
    assert out['q1_mz'].iloc[0] == pytest.approx(out['precursor_mz'].iloc[0])
    assert out['q1_mz'].iloc[0] == pytest.approx(1092.4304, abs=1e-3)


def test_q1_most_abundant_is_two_isotopes_up_for_this_38mer():
    df = _frame('Phospho@S', '27')
    out = precursor_q1_mz(df, isotope='most_abundant')
    assert out['q1_isotope_index'].tolist() == [2]
    assert out['q1_mz'].iloc[0] == pytest.approx(1092.9321, abs=1e-3)


def test_q1_most_abundant_slices_from_mono_isotope_idx_for_a_labelled_peptide():
    core.raw.register_sil_modifications()
    heavy = precursor_q1_mz(_frame('Phospho@S;Label:13C(3)15N(1)@A', '27;38'),
                            isotope='most_abundant')
    light = precursor_q1_mz(_frame('Phospho@S', '27'), isotope='most_abundant')
    assert heavy['q1_isotope_index'].iloc[0] == light['q1_isotope_index'].iloc[0] == 2
    assert heavy['precursor_mz'].iloc[0] == pytest.approx(1093.4322, abs=1e-3)
    assert heavy['q1_mz'].iloc[0] == pytest.approx(1093.9339, abs=1e-3)
    assert (heavy['precursor_mz'].iloc[0] - light['precursor_mz'].iloc[0]) * 4 \
        == pytest.approx(4.00710, abs=1e-4)


def test_q1_isotope_choice_is_validated():
    with pytest.raises(ValueError, match='monoisotope'):
        precursor_q1_mz(_frame('Phospho@S', '27'), isotope='apex')


def test_exported_at_top_level():
    assert core.write_sciex_mrmhr_table is write_sciex_mrmhr_table
    assert core.sciex_mrmhr_table is sciex_mrmhr_table
    assert core.precursor_q1_mz is precursor_q1_mz


class TestEadScheduledLayout:

    @staticmethod
    def _targets():
        return [{'compound_id': f'e{ke:g}', 'precursor_mz': 1093.9339,
                 'rt_min': 5.43, 'electron_ke_ev': float(ke)}
                for ke in (0, 7, 12, 15, 17, 20)]

    def test_column_order_is_literal(self):
        assert SCIEX_MRMHR_LAYOUTS['ead_scheduled'] == (
            'Compound ID', 'Group name', 'Precursor Ion (Da)',
            'TOF Start Mass (Da)', 'TOF Stop Mass (Da)', 'Reaction time (ms)',
            'Accumulation Time (sec)', 'QJet DP (V)', 'Collision energy (V)',
            'CE Spread (V)', 'Fragmentation mode', 'Electron KE (eV)',
            'ETC (%)', 'Retention time (min)',
            'Retention time tolerance (sec)', 'Time bins to sum')

    def test_electron_columns_sit_after_ce_spread(self):
        cols = SCIEX_MRMHR_LAYOUTS['ead_scheduled']
        i = cols.index('CE Spread (V)')
        assert cols[i + 1:i + 4] == ('Fragmentation mode', 'Electron KE (eV)',
                                     'ETC (%)')

    def test_unscheduled_ead_has_them_too_and_no_rt(self):
        cols = SCIEX_MRMHR_LAYOUTS['ead']
        assert 'Electron KE (eV)' in cols
        assert not any(c.startswith('Retention time') for c in cols)

    def test_per_row_electron_energy_sweeps(self):
        t = sciex_mrmhr_table(self._targets(), layout='ead_scheduled',
                              collision_energy=12, fragmentation_mode='EAD',
                              etc_percent=100.0, time_bins_to_sum=8,
                              reaction_time_ms=15.0)
        assert t['Electron KE (eV)'].tolist() == [
            '0.000', '7.000', '12.000', '15.000', '17.000', '20.000']
        assert set(t['Fragmentation mode']) == {'EAD'}
        assert set(t['ETC (%)']) == {'100.0'}
        assert set(t['Time bins to sum']) == {'8'}

    def test_zero_electron_energy_is_written_not_blanked(self):
        t = sciex_mrmhr_table(self._targets(), layout='ead_scheduled',
                              collision_energy=12, fragmentation_mode='EAD')
        assert t['Electron KE (eV)'].iloc[0] == '0.000'

    def test_electron_columns_left_empty_when_not_given(self):
        t = sciex_mrmhr_table(
            [{'compound_id': 'a', 'precursor_mz': 1000.0, 'rt_min': 5.0}],
            layout='ead_scheduled', collision_energy=12)
        for col in ('Fragmentation mode', 'Electron KE (eV)', 'ETC (%)',
                    'Time bins to sum'):
            assert t[col].iloc[0] == ''

    def test_group_is_constant_by_default(self):
        t = sciex_mrmhr_table(self._targets(), layout='ead_scheduled',
                              collision_energy=12, group='Group 1')
        assert set(t['Group name']) == {'Group 1'}


class TestQ1ApexForLabelledPeptides:

    HEAVY = dict(
        sequence='LASVSVSR',
        mods=('Label:13C(6)15N(1)@L;Label:13C(3)15N(1)@A;'
              'Label:13C(3)15N(1)@S;Label:13C(5)15N(1)@V;'
              'Label:13C(5)15N(1)@V;Label:13C(6)15N(4)@R'),
        mod_sites='1;2;4;5;6;8', charge=2)

    def _frame(self, **over):
        import pandas as pd
        row = dict(self.HEAVY)
        row.update(over)
        return pd.DataFrame([row])

    def test_apex_of_a_heavily_labelled_peptide_is_its_monoisotope(self):
        import zenotof8600_analytics as core
        core.raw.register_sil_modifications()
        q = core.precursor_q1_mz(self._frame(), isotope='most_abundant')
        assert int(q['q1_isotope_index'].iloc[0]) == 0
        assert q['q1_mz'].iloc[0] == pytest.approx(
            q['precursor_mz'].iloc[0], abs=1e-6)

    def test_unlabelled_peptide_is_unaffected(self):
        import zenotof8600_analytics as core
        q = core.precursor_q1_mz(
            self._frame(mods='', mod_sites=''), isotope='most_abundant')
        assert int(q['q1_isotope_index'].iloc[0]) == 0

    def test_a_long_peptide_still_picks_a_later_isotope(self):
        import zenotof8600_analytics as core
        q = core.precursor_q1_mz(
            self._frame(sequence='NEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA',
                        mods='', mod_sites='', charge=4),
            isotope='most_abundant')
        assert int(q['q1_isotope_index'].iloc[0]) == 2

    def test_monoisotope_mode_is_always_offset_zero(self):
        import zenotof8600_analytics as core
        core.raw.register_sil_modifications()
        q = core.precursor_q1_mz(self._frame(), isotope='monoisotope')
        assert int(q['q1_isotope_index'].iloc[0]) == 0
