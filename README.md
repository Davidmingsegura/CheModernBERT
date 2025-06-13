# CheModernBERT

A ModernBERT-based encoder for chemistry: continuous pre-training on SMILES-augmented scientific corpora, long-context representation learning, and uncertainty-aware downstream predictions.

## Overview

CheModernBERT is a bidirectional transformer encoder specialized for multi-purpose applications. It combines natural language understanding and molecular representation learning by injecting SMILES strings into scientific text corpora and leveraging extended context lengths. The model provides versatile representations suitable for diverse downstream tasks—classification, regression, retrieval, clustering—and supports uncertainty-aware predictions using Gaussian Process (GP) and Variational GP (VGP) surrogates.

## Table of Contents

## Motivation

Transformer encoder models (e.g., BERT, SciBERT, ChemBERTa ...) have transformed NLP by capturing bidirectional context via self-supervision. When applied to chemistry, SMILES strings serve as “text-like” molecular representations for self-supervised learning. However, standalone SMILES models may lack broader scientific context, and typical context windows limit incorporation of long documents (e.g., full articles or multi-step procedures). CheModernBERT addresses these gaps by:

* Injecting SMILES inline into scientific text corpora to jointly learn language and molecular semantics.
* Leveraging extended context (up to 8192 tokens) from ModernBERT architecture to handle full articles or procedural text with many SMILES inline.
* Conducting large-scale continuous pretraining on diverse corpora to produce versatile molecular and text representations.
* Enabling downstream tasks across classification, regression, retrieval, clustering, and uncertainty estimation.

## Main Contributions

1. **SMILES-augmented scientific corpora and pipeline**: Assembled and processed large text corpora from scientific articles, educational web content, and reaction procedures. Automated identification of chemical mentions, lookup and canonicalization of SMILES, and inline injection into text. Supports configurable injection density for ablations.

2. **Extended-context ModernBERT pretraining**: Built on ModernBERT with context length up to 8192 tokens using rotary embeddings, GeGLU layers, bias-free projections, flash attention, and unpadding to efficiently process long sequences. Pretrained with Masked Language Modeling (MLM) over mixed SMILES-text corpora.

3. **Comprehensive benchmarking**: Evaluated the model on SMILES-based classification/regression tasks (e.g., property prediction), scientific text tasks, retrieval tasks via contrastive fine-tuning, clustering analyses, and uncertainty-aware predictions using GP/VGP surrogates on frozen embeddings.

4. **Open-source codebase**: Modular scripts for data preprocessing, pretraining, embedding extraction, and downstream experiments. Notebooks for analysis and visualization.

## Data and Preprocessing

### Corpora

* **Scientific articles**: Collections of open-access chemistry articles processed into plain text.
* **Educational web content**: Filtered chemistry-related educational text from web crawls.
* **Reaction procedures**: Experimental procedures from patent data paired with reaction SMILES.

### SMILES Injection Pipeline

1. Text extraction and normalization (e.g., PDF-to-text via Grobid/Nougat).
2. Sentence splitting.
3. Chemical entity recognition (e.g., CDE2) to identify IUPAC names, formulas, trivial names.
4. SMILES lookup and canonicalization (e.g., OPSIN, PubChemPy, RDKit).
5. Inline injection or wrapping of SMILES tokens within text with special markers.
6. Configurable injection density for ablations.
7. Dataset splitting for pretraining (e.g., train/validation).

## Model Architecture

CheModernBERT extends ModernBERT with:

* **Extended Context (up to 8192 tokens)**: Allows processing of long sequences such as full articles or multi-step procedures.
* **Rotary Positional Embeddings (RoPE)**: Encodes relative positions directly in attention computation.
* **GeGLU Feed-Forward Layers**: GELU-activated gating for expressivity.
* **Bias-free Projections**: Reducing parameter count and memory footprint.
* **Flash Attention & Unpadding**: Efficient attention and skipping padded tokens to save compute.
* **Mixed-Precision Training (BF16)** and Distributed Data Parallel for scalability.

## Pretraining Setup

* **Masked Language Modeling (MLM)**: Standard MLM objective on sequences up to 8192 tokens, masking 15% of tokens with dynamic unpadding.
* **Training Details**: Mixed-precision BF16, multi-GPU distributed training, appropriate learning rates and optimizers, checkpointing and logging (e.g., Weights & Biases).
* **Corpora Mix**: Combined SMILES-injected scientific text, educational content, and procedural text for broad coverage.
* **Optional Contrastive Pretraining**: For retrieval tasks, InfoNCE loss on paired SMILES-text or text-text examples.

## Evaluation and Benchmarking

* **Classification & Regression**: Extract CLS embeddings for SMILES or text inputs; train surrogate models (e.g., Random Forest, GP) to predict properties or classes; evaluate metrics such as F1, MAE, R².
* **Retrieval**: Contrastive fine-tuning on paired datasets (SMILES ↔ description, query ↔ passage); evaluate Top-1 accuracy, MRR, Recall\@k.
* **Clustering**: Embedding extraction, dimensionality reduction (e.g., t-SNE) for visualization, K-means clustering with Silhouette score and V-measure.
* **Uncertainty-Aware Predictions**: Use frozen CheModernBERT embeddings as features for GP/VGP models to obtain predictive uncertainties; evaluate calibration and accuracy.

## Uncertainty-Aware Predictions

* **Approach**: Feed frozen CLS embeddings into GP or VGP models for regression or classification to obtain predictive distributions.
* **Metrics**: Calibration error, reliability diagrams, predictive accuracy compared to baselines.

## Repository Structure & Usage

```
CheModernBERT/
├── README.md
├── ...
```

### Dependencies


### Data Preparation

1. Obtain raw data: download or access scientific articles, educational text dumps, reaction procedure data.
2. Convert to plain text and clean.
3. Identify chemical mentions and lookup SMILES; canonicalize with RDKit.
4. Inject SMILES inline with special tokens; control density for ablation.
5. Split into train/validation for pretraining; save in consumable format (e.g., JSONL).

### Pretraining

* Use HuggingFace Trainer or custom training loops with PyTorch DDP.
* Configure ModernBERT with extended context length, RoPE, GeGLU, flash attention.
* Set MLM objective over long sequences with dynamic unpadding.
* Train with BF16 precision, multiple GPUs, appropriate learning rate schedules.
* Monitor training and validation loss; run for sufficient epochs (e.g., \~60 epochs) to converge.

### Extracting Embeddings

* Load pretrained CheModernBERT in evaluation mode.
* For each input (SMILES or text), tokenize and process to extract CLS token embedding.
* Batch appropriately for long inputs; save embeddings as NumPy arrays or tensors with identifiers.

### Downstream Experiments

#### Classification & Regression

* Prepare datasets: inputs and labels.
* Extract embeddings via CheModernBERT.
* Train surrogate models (Random Forest, XGBoost, GP) on frozen embeddings.
* Evaluate metrics: F1, MAE, R²; compare against baselines and ablations.

#### Retrieval (Contrastive Fine-tuning)

* Prepare paired data (e.g., SMILES-description, query-passage).
* Fine-tune CheModernBERT with InfoNCE loss.
* Build embedding for retrieval applicattions
* Evaluate Top-1 accuracy, MRR, Recall\@k.

#### Clustering

* Extract embeddings for a labeled dataset.
* Use t-SNE for 2D visualization.
* Apply K-means clustering; compute Silhouette score and V-measure against true labels.

#### Uncertainty with GP/VGP

* Use frozen embeddings as features.
* Train GP or VGP models for regression/classification.
* Evaluate predictive mean, variance; plot calibration curves; compute calibration error.
* Optionally explore joint embedding fine-tuning via GP marginal likelihood.

### Usage Examples


## Results Summary

* **SMILES-augmented pretraining** improves molecular and text task performance compared to text-only or SMILES-only models.
* **Extended context** allows handling long documents and complex procedures in a single pass.
* **Versatile embeddings**: Effective for classification, regression, retrieval, clustering, and uncertainty calibration.
* **Open-source and modular**: Scripts and notebooks provided for reproducibility and extension.

## Citation

This README.md was made with help of ChatGPT.

## License

Specify an open-source license (e.g., MIT License) in a separate `LICENSE` file.
