# Cobbled together from the out-of-date miniconda3 Dockerfile
# https://github.com/ContinuumIO/docker-images/blob/7200d37470fdac2ea55390f9af138263ce868378/miniconda3/Dockerfile
# and ilastik developer installation instructions
# https://github.com/ilastik/ilastik-build-conda/blob/40fd167fc66069ca8dcd03b659a707685e8ddca0/README.md

# todo: handle getting project files in
# todo: handle output dir
# todo: handle credentials, args etc.

FROM debian:8.5

MAINTAINER Chris Barnes <barnesc@janelia.hhmi.org>

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git && \
    rm -rf /var/lib/apt/lists/*

###############
# setup conda #
###############

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda config --set changeps1 no && \
    /opt/conda/bin/conda update -q conda

ENV PATH=/opt/conda/bin:$PATH

#################
# setup ilastik #
#################

RUN conda create --yes --name ilastik-dev ilastik-dependencies-no-solvers=1.3 --channel ilastik-forge --channel conda-forge && \
    rm -rf /opt/conda/pkgs/*
# todo: cplex/gurobi?

ENV CONDA_DEFAULT_ENV=ilastik-dev \
    CONDA_PREFIX=/opt/conda/envs/ilastik-dev

###########################
# setup skeleton_synapses #
###########################

COPY requirements/prod.txt /skeleton_synapses/requirements/prod.txt

RUN pip install --trusted-host pypi.python.org -U pip==9.0.1 && \
    pip install --trusted-host pypi.python.org -r requirements/prod.txt

WORKDIR /skeleton_synapses

COPY . /skeleton_synapses

# install other dependencies and run unit tests
RUN pip install --trusted-host pypi.python.org -r requirements/test.txt && \
    make test

ENV PATH=/skeleton_synapses/bin:$PATH

# expose the progress monitor
EXPOSE 8088

# run skeleton_synapses when the container starts
ENTRYPOINT ["/bin/bash", "/skeleton_synapses/bin/skeleton_synapses"]

# if no overriding arguments are passed, print help message
CMD ["--help"]
