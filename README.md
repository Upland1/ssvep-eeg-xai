# ssvep-eeg-xai

This repository is being developed in stages.

Current focus:
- loading and validating EBR recordings
- band-pass filtering (1–60 Hz)
- artifact rejection using Vp-p and standard deviation thresholds
- condition-based sub-windowing using the MARK channel
- PSD extraction and feature matrix construction for SSVEP classification

This project is intentionally kept minimal at this stage so the workflow follows the original analysis scripts step by step.
