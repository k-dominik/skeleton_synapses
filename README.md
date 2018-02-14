[![Build Status](https://travis-ci.org/clbarnes/skeleton_synapses.svg?branch=master)](https://travis-ci.org/clbarnes/skeleton_synapses)
[![Coverage Status](https://coveralls.io/repos/github/clbarnes/skeleton_synapses/badge.svg?branch=master)](https://coveralls.io/github/clbarnes/skeleton_synapses?branch=master)

WORK IN PROGRESS

# skeleton_synapses

A utility for automatically detecting synapses, compatible with [CATMAID](https://catmaid.readthedocs.io/en/stable/)
and [synapsesuggestor](https://github.com/clbarnes/CATMAID-synapsesuggestor).

## INSTALLATION:

N.B. conda must be version <= 4.3 to work with pyenv

- Use `conda` to [install `ilastik`](https://github.com/ilastik/ilastik-build-conda/blob/master/README.md),
 and then `pip` to install the requirements in `./requirements/`.
- `cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/`
- Use `pip install -r requirements/prod.txt` for production installations,
add `-r requirements/test.txt` to run unit tests, and
add `-r requirements/travis.txt` for continuous integration and coverage
with [travis](https://travis-ci.org/) and [coveralls](https://coveralls.io/).


## USAGE:

```bash
usage: skeleton_synapses [-h] [-o OUTPUT_DIR] [-r ROI_RADIUS_PX] [-f FORCE]
               credentials-path stack-id input-file-dir skeleton-ids
               [skeleton-ids ...]

positional arguments:
  credentials-path      Path to a JSON file containing CATMAID credentials
                        (see credentials/example.json)
  stack-id              ID or name of image stack in CATMAID
  input-file-dir        A directory containing project files.
  skeleton-ids          Skeleton IDs in CATMAID

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        A directory containing output files
  -r ROI_RADIUS_PX, --roi-radius-px ROI_RADIUS_PX
                        The radius (in pixels) around each skeleton node to
                        search for synapses
  -f FORCE, --force FORCE
                        Whether to delete all prior results for a given
                        skeleton: pass 1 for true or 0
```


### Example:

```bash
bash bin/skeleton_synapses \
    credentials/my_creds.json \
    1 \
    ~/my_project_files/ \
    123456 \
    654321 \
    789012 \
    -o ~/my_output_dir/ \
    -r 150 \
    -f 0
```


## TESTING:

### Unit tests

```bash
pytest
```

A running CATMAID instance is not required; nor is a functioning installation of ilastik. Tests only cover the python
 code surrounding those two interfaces.

### Integration tests

```bash
pytest tests/integration_tests --input_dir path/to/input/ --credentials_path path/to/credentials.json
```

This requires ilastik, valid project files, and a running instance of CATMAID.


## CATMAID CREDENTIALS:

See `./credentials/example.json` for the credentials file required by `skeleton_synapses`.


## ENVIRONMENT VARIABLES:

Environment variables control `skeleton_synapses`' utilisation of compute resources.
Store different environment variable configurations in `./env_vars/` (see `example`).
Source these environment variables with `./bin/set_env_vars <file_name>`
