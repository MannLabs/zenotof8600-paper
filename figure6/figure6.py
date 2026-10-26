"""Figure 6 -- targeted quantification on the 8600."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import figure_style as fs

LETTER = {'standard': 'a', 'calibration_curve': 'b', 'accuracy': 'c',
          'precision': 'd', 'artwork': 'e', 'volcano': 'f',
          'occupancy': 'g', 'cid_spectrum': 'h', 'ead_spectrum': 'i'}


def main():
    fs.init()
    import calibration
    import standard
    import volcano
    import spectra
    import occupancy

    print(f'=== panel {LETTER["standard"]}: the Promega 6x5 standard ===')
    standard.main(LETTER['standard'])
    print('\n=== panels b, c, d: calibration, accuracy, precision ===')
    calibration.main(LETTER['calibration_curve'], LETTER['accuracy'],
                     LETTER['precision'])
    print(f'\n=== panel {LETTER["volcano"]}: PFF volcano, ZT Scan ===')
    volcano.main(LETTER['volcano'])
    print(f'\n=== panels {LETTER["cid_spectrum"]}, '
          f'{LETTER["ead_spectrum"]}: CID and EAD ===')
    spectra.main(LETTER['cid_spectrum'], LETTER['ead_spectrum'])
    print(f'\n=== panel {LETTER["occupancy"]}: pS129 occupancy, MRM-HR ===')
    occupancy.main(LETTER['occupancy'])

    fs.report_printed_sizes()


if __name__ == '__main__':
    main()
