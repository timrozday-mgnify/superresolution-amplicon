# sra-skiver: skiver Rust binary (`skiver dump`) + Python env for error-model
# training, mis-mapping (incl. edlib sampling), and Pyro composition inference.
# Build context is the repo root so the vendor/skiver submodule is visible:
#   docker build -f containers/skiver/Dockerfile -t sra-skiver .

# ---- stage 1: build the skiver Rust binary --------------------------------
FROM rust:1-bookworm AS builder

# rust-htslib links the system htslib and uses libclang for bindgen.
# libhts-dev pulls the curl dev package it needs (gnutls variant) itself, so we
# don't pin libcurl*-dev here (avoids the gnutls/openssl conflict).
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang libclang-dev pkg-config cmake make \
        libhts-dev zlib1g-dev libbz2-dev liblzma-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY vendor/skiver/ ./
RUN cargo build --release --bin skiver && cp target/release/skiver /usr/local/bin/skiver

# ---- stage 2: runtime -----------------------------------------------------
FROM python:3.11-slim-bookworm

ARG QUARTO_VERSION=1.5.57

# htslib shared library for the dynamically linked binary (apt resolves the
# matching libcurl runtime dep); procps provides `ps`, required by Nextflow;
# curl fetches the Quarto .deb (error-model report rendering).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhts3 procps curl \
    && curl -fsSL -o /tmp/quarto.deb \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-$(dpkg --print-architecture).deb" \
    && dpkg -i /tmp/quarto.deb \
    && rm -f /tmp/quarto.deb \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/skiver /usr/local/bin/skiver

# skiver's error-model library + training/calibration scripts. subspecies_infer.py
# imports `lib.error_application` from here (SKIVER_SCRIPTS is on sys.path).
ENV SKIVER_SCRIPTS=/opt/skiver/scripts
COPY vendor/skiver/scripts/ ${SKIVER_SCRIPTS}/

# Per-platform Quarto error-model report, rendered by SKIVER_REPORT.
ENV SKIVER_NOTEBOOKS=/opt/skiver/notebooks
COPY vendor/skiver/notebooks/context_error_model_*.qmd vendor/skiver/notebooks/_context_error_model_body.qmd ${SKIVER_NOTEBOOKS}/

# CPU-only torch keeps the image lean. pyro drives inference; edlib is the argmax
# assigner for the sampling mis-mapping path; plotly/ipykernel/papermill render the
# Quarto error-model report. The pipeline's own bin/ scripts are staged onto PATH by
# Nextflow, so they are not baked in here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        numpy scipy pandas h5py pyyaml matplotlib rich "pyro-ppl>=1.9" \
        edlib plotly ipykernel jupyter papermill \
    && python -m ipykernel install --name skiver --sys-prefix

# Nextflow's docker executor runs the container as the host UID, not root, so
# $HOME (unset for that UID) must point somewhere writable for Quarto/Deno's
# cache dir (~/.cache/quarto) and Jupyter's runtime dir.
ENV HOME=/tmp
RUN chmod 1777 /tmp

RUN skiver --version && quarto --version && python -c "import edlib, pyro, torch"

# The checks above ran as root and left root-owned junk under /tmp that a later
# non-root runtime UID can't write around; clear it so /tmp starts empty.
RUN rm -rf /tmp/.cache /tmp/tmp*
