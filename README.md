# NLP LLM Project - Email Routing with LLMs

Automatic email routing system using Large Language Models for customer support ticket classification.

## Project Overview

This project develops and compares three types of LLM-based agents for automatically routing incoming customer support emails/tickets to appropriate departments:

1. **Agent 1**: Frozen GPT-2/DistilGPT2 with prompting (zero-shot classification)
2. **Agent 2**: GPT-2/DistilGPT2 with LoRA fine-tuning (parameter-efficient training)
3. **Agent 3**: DistilBERT discriminative classifier (full fine-tuning)

### Target Departments
- Technical Support
- Customer Service
- Billing and Payments
- Sales and Pre-Sales
- General Inquiry

### Dataset
**Tobi-Bueck/customer-support-tickets** from Hugging Face
- English-only tickets
- 5-department classification
- Split: 80% train / 10% validation / 10% test

## Project Structure

```
nlp-llm-project/
├── README.md                           # Project documentation
├── PROJECT_TODO.md                     # Detailed task tracking (30 tasks)
├── LICENSE                             # Project license
├── .gitignore                          # Git ignore patterns
├── build-project-conda-environment.yml # Conda environment specification
├── src/                                # Source code
│   └── datapreparation.py             # Data loading and preprocessing
├── notebooks/                          # Jupyter notebooks
│   └── notebook.ipynb                 # Main project notebook
├── docs/                               # Documentation and lab materials
│   ├── nlp-labs/                      # Course lab materials
│   └── nlp-notes/                     # Course notes
├── data/                               # Data directory (auto-downloaded)
├── models/                             # Saved model checkpoints
├── results/                            # Evaluation results and visualizations
└── assignment/                         # Assignment materials
```

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create environment from yml file
conda env create -f build-project-conda-environment.yml

# Activate environment
conda activate build-nanogpt

# Install as Jupyter kernel
python -m ipykernel install --user --name build-nanogpt --display-name "build-nanogpt"
```

### 2. Verify Installation

```bash
# Test data preparation
python -c "from src.datapreparation import load_and_prepare_data; train_ds, val_ds, test_ds, _, _, _ = load_and_prepare_data(); print(f'✅ Setup complete! Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')"
```

### 3. Open Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/notebook.ipynb
```

Or open in VSCode with the `build-nanogpt` kernel selected.

## Environment Details

- **Python**: 3.10.19
- **Key Libraries**:
  - PyTorch 2.9.1
  - Transformers 4.57.3
  - PEFT 0.18.0 (LoRA implementation)
  - Datasets 4.4.1
  - scikit-learn 1.7.2

## Project Phases

### Phase 1: Setup & Environment ✅
- [x] Conda environment created and tested
- [x] Data preparation module implemented
- [x] Jupyter notebook structure created
- [x] All dependencies installed

### Phase 2: Agent 1 - Frozen Models with Prompting
- [ ] Load and test GPT-2/DistilGPT2 models
- [ ] Design and experiment with prompt templates
- [ ] Implement inference and evaluation

### Phase 3: Agent 2 - Fine-tuning with LoRA
- [ ] Configure LoRA for parameter-efficient training
- [ ] Prepare training data
- [ ] Fine-tune GPT-2/DistilGPT2
- [ ] Evaluate and compare with Agent 1

### Phase 4: Agent 3 - DistilBERT Classifier
- [ ] Load DistilBERT for sequence classification
- [ ] Fine-tune discriminative model
- [ ] Comprehensive evaluation

### Phase 5: Comprehensive Evaluation
- [ ] Compare all model variants
- [ ] Generate visualizations and metrics
- [ ] Analyze computational efficiency

### Phase 6: Documentation & Deliverables
- [ ] Write technical report
- [ ] Prepare presentation materials
- [ ] Final submission

## Usage

### Running the Main Notebook

1. Open `notebooks/notebook.ipynb` in Jupyter or VSCode
2. Select the `build-nanogpt` kernel
3. Run cells sequentially

### Data Loading

```python
from src.datapreparation import load_and_prepare_data

# Load preprocessed data
train_ds, val_ds, test_ds, label_list, label2id, id2label = load_and_prepare_data()
```

## Model Architectures

| Model | Parameters | Layers | Hidden Size | Attention Heads |
|-------|------------|--------|-------------|-----------------|
| GPT-2 | 124M | 12 | 768 | 12 |
| DistilGPT2 | 82M | 6 | 768 | 12 |
| DistilBERT | 66M | 6 | 768 | 12 |

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Confusion Matrix**: Detailed classification breakdown
- **Inference Time**: Average time per sample
- **Training Time**: Total time for fine-tuned models
- **Memory Usage**: Peak memory during inference

## Key Resources

- **Lab 1**: GPT-2 model usage and text generation
- **Lab 5**: LoRA theory and parameter-efficient fine-tuning
- **Lab 6**: Training loop implementation
- **Assignment PDF**: Project specifications and evaluation criteria

## Expected Results

- **Agent 1** (Zero-shot): Fast inference, moderate accuracy
- **Agent 2** (LoRA): Improved accuracy, efficient training
- **Agent 3** (DistilBERT): Best accuracy, standard fine-tuning

## Contributing

This is an academic project. For questions or issues, please refer to the PROJECT_TODO.md file.

## License

See LICENSE file for details.

## Acknowledgments

- Course: NLP with Large Language Models
- Dataset: Tobi-Bueck/customer-support-tickets (Hugging Face)
- Frameworks: Hugging Face Transformers, PEFT
