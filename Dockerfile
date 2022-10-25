FROM mambaorg/micromamba:latest
MAINTAINER William E Fondrie <fondriew@gmail.com>
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER dist/*.whl /tmp/

WORKDIR /app

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN WHEEL=$(ls /tmp/*.whl) && pip install ${WHEEL}
