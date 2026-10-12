# sramp-python: the Python environment for every step that does not need skiver --
# amplicon extraction, matrix keys, panel preparation, read conversion, composition
# inference and its fit check. Training, simulation and the alignment kernel (whose
# `--align_distance_decay auto` samples a skiver error model) stay in sra-skiver.
#
# No Rust build, so it builds for arm64 as well as amd64: an Apple-silicon laptop runs
# it natively under Docker instead of emulating x86, which made inference ~4.5x slower.
#   docker build -f containers/python/Dockerfile -t sramp-python .
FROM python:3.11-slim-bookworm

# procps provides `ps`, which Nextflow needs.
RUN apt-get update && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image lean. pyro drives inference; edlib trims primers in
# reads_to_fasta.py. edlib ships no arm64 wheel, so g++ is installed to build it and
# removed in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scipy pandas "pyro-ppl>=1.9" edlib \
    && apt-get purge -y --auto-remove g++ && rm -rf /var/lib/apt/lists/*

# Nextflow's docker executor runs the container as the host UID, so $HOME must be
# somewhere writable (torch's cache).
ENV HOME=/tmp
RUN chmod 1777 /tmp

RUN python -c "import torch, pyro, edlib, scipy, pandas" && rm -rf /tmp/.cache /tmp/tmp*
