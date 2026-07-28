# Multilingual Probabilistic Tokenization Research

This local research project compares two multilingual language-model pipelines for English, Spanish, and Hindi:

- **Control:** a BPE tokenizer trained from scratch with a randomly initialized XLM-R-base-style model.
- **Proposed:** a language-aware probabilistic Unigram tokenizer that generates multiple candidate tokenizations, aligns them by character spans, and fuses their representations before the same XLM-R-base-style model.

Both systems use the same CulturaX data split, training objective, model configuration, optimizer, schedule, and downstream GLueCoS evaluation protocol. The experiment tests whether language-aware tokenization uncertainty improves downstream performance.

## Run the project

Create and activate a local Python environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cuda130.txt
pip install -r requirements.txt
jupyter lab
```

For a CPU-only or different CUDA installation, install the appropriate PyTorch build first, then install `requirements.txt`.

Run the notebooks in order:

1. `notebooks/01_environment_setup.ipynb` — validate the environment and configuration.
2. `notebooks/02_download_datasets.ipynb` — download and save CulturaX and GLueCoS locally. Set `HF_TOKEN` first and accept the CulturaX Hugging Face terms.
3. `notebooks/03_research_pipeline.ipynb` — prepare data, train both tokenizers, and pretrain both MLMs.
4. `notebooks/04_gluecos_evaluation.ipynb` — fine-tune both selected MLM checkpoints and save GLueCoS predictions.
5. `notebooks/07_experiment_summary_and_statistics.ipynb` — generate paired statistical tests and the experiment summary.

All behavior is configured in `configs/config.yaml`. Set `runtime.mode` to `train`, `resume`, or `evaluate` before running a stage. Generated data, models, checkpoints, logs, and results remain local and are ignored by Git.
