"""The developer-mode minimum EAD cell fill, against the scheduled default."""
import annotate_background as ab

PANELS = (('standard only', 'A6'),
          ('HEK, default', 'B1'),
          ('HEK, dev-mode min fill', 'B9'),
          ('HEK, 5 ms load time', 'B10'))


if __name__ == '__main__':
    ab.main(panels=PANELS, name='devmode')
