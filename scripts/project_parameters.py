from pathlib import Path

# Paths
paths = {'main': Path.cwd()}
paths.update({'data': Path.joinpath(paths.get('main'), 'data'),
              'output': Path.joinpath(paths.get('main'), 'output'),
              'scripts': Path.joinpath(paths.get('main'), 'scripts')})