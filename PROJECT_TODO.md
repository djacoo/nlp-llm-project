# NLP LLM Project - Email Routing with LLMs
## Project TODO List

---

## Project Overview

Develop three LLM-based agents for automatic email routing to departments:
1. **Agent 1**: Frozen GPT-2/DistilGPT2 with prompting (zero-shot)
2. **Agent 2**: GPT-2/DistilGPT2 with LoRA fine-tuning
3. **Agent 3**: DistilBERT discriminative classifier

**Target Departments**: Technical Support, Customer Service, Billing and Payments, Sales and Pre-Sales, General Inquiry

**Dataset**: Tobi-Bueck/customer-support-tickets (Hugging Face)

---

## Resource Mapping

| Agent/Phase | Primary Resources | Files to Use |
|-------------|------------------|--------------|
| **Setup** | Assignment PDF Section 8 | datapreparation.py code (provided) |
| **Agent 1** | Lab 1 - GPT-2 model usage | play_Template.ipynb |
| **Agent 2** | Lab 5 (LoRA theory) + Lab 6 (training) | Lab3_LLM-Fine-Tuning.pdf, train_gpt2_step0-5.py |
| **Agent 3** | Transformers docs | DistilBERT sequence classification |
| **Evaluation** | Lab 6 + Assignment Section 5 | Metrics calculation code |

---

## TODO List (30 Tasks)

### Phase 1: Setup & Environment (Tasks 1-4) ✅ COMPLETED

- [x] **Task 1**: Setup - Create project structure and prepare conda environment ✅
  - ✅ Created `build-nanogpt` conda environment (Python 3.10.19)
  - ✅ PyTorch 2.9.1, Transformers 4.57.3, datasets 4.4.1 installed
  - ✅ Additional packages: peft 0.18.0, evaluate 0.4.6, scikit-learn 1.7.2
  - ✅ Jupyter kernel registered and working
  - ✅ Professional directory structure created (src/, notebooks/, docs/, models/, results/, data/)

- [x] **Task 2**: Setup - Create datapreparation.py with code from assignment PDF ✅
  - ✅ Code implemented from Assignment PDF Section 8
  - ✅ Location: `src/datapreparation.py` (reorganized from project root)
  - ✅ RANDOM_SEED = 42 verified
  - ✅ Function `load_and_prepare_data()` working correctly

- [x] **Task 3**: Setup - Test datapreparation.py ✅
  - ✅ Successfully loads Tobi-Bueck/customer-support-tickets dataset
  - ✅ English-only filter working (28,261 from 61,765 total)
  - ✅ 5 departments filtered correctly (16,562 tickets)
  - ✅ 80/10/10 train/val/test split confirmed (13,249 / 1,656 / 1,657)
  - ✅ Label distribution: Technical Support (6,476), Customer Service (3,471), Billing (2,307), Sales (655), General (340)

- [x] **Task 4**: Setup - Create main Jupyter notebook structure ✅
  - ✅ File: `notebooks/notebook.ipynb` (reorganized)
  - ✅ All sections created: Data Loading, Agent 1, Agent 2, Agent 3, Evaluation, Results, Conclusions
  - ✅ Import updated: `from src.datapreparation import load_and_prepare_data`
  - ✅ Notebook tested and working with build-nanogpt kernel
  - ✅ Professional README.md, .gitignore, requirements.txt, setup.py created

---

### Phase 2: Agent 1 - Frozen Models with Prompting (Tasks 5-9)

**Resource**: Lab 1 (play_Template.ipynb)

- [ ] **Task 5**: Agent 1 - Load frozen GPT-2 model for email routing
  - Adapt Lab 1 code: `GPT2LMHeadModel.from_pretrained("gpt2")`
  - Set model to eval mode: `model.eval()`
  - No weight updates (frozen model)

- [ ] **Task 6**: Agent 1 - Design instruction prompts for classification
  - Experiment with 3-5 prompt templates:
    - Template 1: "Classify this customer support email into one of these departments: [list]. Email: [text]. Department:"
    - Template 2: "You are a customer support router. Read this email and output only the department name: [text]"
    - Template 3: (design your own variations)
  - Document all prompts tested

- [ ] **Task 7**: Agent 1 - Implement inference function for department extraction
  - Generate text with model
  - Parse output to extract department name
  - Map to one of 5 valid departments
  - Handle edge cases (invalid outputs, partial matches)

- [ ] **Task 8**: Agent 1 - Repeat for DistilGPT2 model
  - Load: `GPT2LMHeadModel.from_pretrained("distilgpt2")`
  - Use same prompts from Task 6
  - Compare output quality with GPT-2

- [ ] **Task 9**: Agent 1 - Evaluate both models on test set
  - Metrics: Accuracy, inference time per sample
  - Test on full test_ds from datapreparation
  - Record best performing prompt for each model
  - Save predictions for confusion matrix later

---

### Phase 3: Agent 2 - Fine-tuning with LoRA (Tasks 10-16)

**Resources**: Lab 5 (LoRA theory), Lab 6 (training code)

- [ ] **Task 10**: Agent 2 - Study LoRA materials
  - Read Lab 5 PDF: Understanding Low-Rank Adaptation
  - Review QLoRA workflow diagram (page 12)
  - Understand parameter efficiency (10,000x reduction)

- [ ] **Task 11**: Agent 2 - Setup LoRA configuration for GPT-2
  - Install: `pip install peft`
  - Import: `from peft import LoraConfig, get_peft_model`
  - Configure LoRA:
    ```python
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        target_modules=["c_attn"],  # attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    ```

- [ ] **Task 12**: Agent 2 - Prepare training data format
  - Format: `"[Instruction prompt] Email: [subject+body] Department: [label]"`
  - Use best prompt from Agent 1 (Task 6)
  - Tokenize with GPT-2 tokenizer
  - Create DataLoader (adapt Lab 6 DataLoaderLite class)

- [ ] **Task 13**: Agent 2 - Implement training loop for GPT-2 with LoRA
  - Adapt Lab 6 training code (train_gpt2_step5.py)
  - Use AdamW optimizer (lr=3e-4)
  - Implement validation evaluation every N steps
  - Save best model checkpoint
  - Target: 3-5 epochs, monitor train/val loss

- [ ] **Task 14**: Agent 2 - Train GPT-2 with LoRA
  - Run training loop
  - Monitor: loss convergence, validation accuracy
  - Stop if validation accuracy plateaus
  - Expected: ~1-2 hours on CPU, faster with GPU

- [ ] **Task 15**: Agent 2 - Evaluate fine-tuned GPT-2 on test set
  - Load best checkpoint
  - Run inference on test_ds
  - Compare with frozen GPT-2 (Task 9)
  - Metrics: accuracy, inference time

- [ ] **Task 16**: Agent 2 - Repeat LoRA fine-tuning for DistilGPT2
  - Same process as Tasks 11-15
  - Use DistilGPT2 as base model
  - Compare with frozen DistilGPT2

---

### Phase 4: Agent 3 - DistilBERT Classifier (Tasks 17-21)

**Resource**: Hugging Face Transformers documentation

- [ ] **Task 17**: Agent 3 - Load DistilBERT for sequence classification
  - Import: `from transformers import DistilBertForSequenceClassification`
  - Load: `DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)`
  - Configure label mapping from datapreparation.py

- [ ] **Task 18**: Agent 3 - Prepare dataset for DistilBERT
  - Tokenize: subject + " " + body
  - Create input_ids, attention_mask
  - Convert labels to numeric IDs (0-4)
  - Use Hugging Face Dataset.map() for efficient processing

- [ ] **Task 19**: Agent 3 - Implement training loop
  - Option A: Use Hugging Face Trainer API (recommended)
  - Option B: Custom training loop (like Lab 6)
  - Training args: lr=2e-5, batch_size=16, epochs=3
  - Use early stopping based on validation accuracy

- [ ] **Task 20**: Agent 3 - Train DistilBERT classifier
  - Run training
  - Monitor validation accuracy each epoch
  - Save best model
  - Expected: 30-60 minutes

- [ ] **Task 21**: Agent 3 - Evaluate DistilBERT on test set
  - Metrics: accuracy, precision, recall, F1 per class
  - Measure inference time and memory usage
  - Generate predictions for confusion matrix

---

### Phase 5: Comprehensive Evaluation (Tasks 22-26)

- [ ] **Task 22**: Evaluation - Calculate metrics for ALL models
  - For each of 5+ models (GPT2, DistilGPT2, fine-tuned versions, DistilBERT):
    - Overall accuracy
    - Per-class precision, recall, F1-score
    - Confusion matrix
  - Use sklearn.metrics: classification_report, confusion_matrix

- [ ] **Task 23**: Evaluation - Measure computational resources
  - Training time (for fine-tuned models)
  - Inference time per sample (average over test set)
  - Memory usage during inference
  - Number of trainable parameters (especially for LoRA)

- [ ] **Task 24**: Results - Create comparison table
  - Format from Assignment Section 5:
    | Model | Accuracy | Precision | Recall | F1 | Train Time | Inference Time |
    |-------|----------|-----------|--------|----|-----------:|---------------:|
  - Include all model variants

- [ ] **Task 25**: Results - Generate visualization charts
  - Accuracy comparison bar chart
  - Confusion matrices (one per model, use seaborn heatmap)
  - Training curves (loss/accuracy vs epochs) for fine-tuned models
  - Inference time comparison

- [ ] **Task 26**: Results - Analyze prompt performance (Agent 1)
  - Compare accuracy across different prompts tested
  - Document best/worst performing prompts
  - Explain why certain prompts work better

---

### Phase 6: Documentation & Deliverables (Tasks 27-30)

- [ ] **Task 27**: Documentation - Write short report (<5 pages)
  - **Section 1**: Introduction & Methodology
    - Brief description of each agent
    - Dataset description and preprocessing
    - Model configurations
  - **Section 2**: Results
    - Comparison table (Task 24)
    - Visualizations (Task 25)
    - Analysis and discussion
  - **Section 3**: Conclusions
    - Best performing model and why
    - Tradeoffs: accuracy vs speed vs complexity
    - Lessons learned
  - Format: PDF, max 5 pages

- [ ] **Task 28**: Deliverables - Export conda environment
  - Command: `conda env export > build-project-conda-environment.yml`
  - Verify file contains all dependencies
  - Test: create new env from yml on different machine

- [ ] **Task 29**: Deliverables - Test notebook on Google Colab
  - Upload notebook.ipynb to Google Drive
  - Open in Colab
  - Adjust paths for Colab environment if needed
  - Run all cells end-to-end
  - Create shareable link (Anyone with link can view)

- [ ] **Task 30**: Deliverables - Prepare final submission
  - **Files to submit** (send 3 days before presentation):
    1. `notebook.ipynb` - Main implementation notebook
    2. `datapreparation.py` - Data loading script
    3. `build-project-conda-environment.yml` - Environment config
    4. `report.pdf` - Short report (<5 pages)
    5. Link to Google Colab notebook
  - Email to: alberto.castellini@univr.it
  - Subject: "NLP LLM Project Submission - [Your Names]"

---

## Key Implementation Notes

### From Lab Materials

1. **Lab 1 Key Patterns**:
   ```python
   # Load model
   model = GPT2LMHeadModel.from_pretrained("gpt2")
   model.eval()

   # Generate with pipeline
   from transformers import pipeline
   generator = pipeline('text-generation', model='gpt2')
   output = generator(prompt, max_length=50)
   ```

2. **Lab 6 Training Pattern**:
   ```python
   # AdamW optimizer
   optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

   # Training loop
   for epoch in range(num_epochs):
       for batch in train_loader:
           optimizer.zero_grad()
           logits, loss = model(x, y)
           loss.backward()
           optimizer.step()
   ```

3. **Lab 5 LoRA Insights**:
   - LoRA adds trainable rank decomposition matrices to frozen weights
   - Reduces trainable parameters by ~10,000x
   - W + ΔW = W + BA (frozen + trainable low-rank)
   - QLoRA adds 4-bit quantization for even lower memory

### Important Configuration Values

- **GPT-2 124M**: 12 layers, 768 hidden, 12 heads, 50257 vocab
- **DistilGPT2 82M**: 6 layers, 768 hidden, 12 heads, 50257 vocab
- **DistilBERT 66M**: 6 layers, 768 hidden, 12 heads, 30522 vocab
- **Random Seed**: 42 (for reproducibility)
- **Dataset Split**: 80% train, 10% validation, 10% test

---

## Success Criteria

✅ **Minimum Requirements**:
- All 3 agent types implemented and working
- Evaluation on test set with accuracy reported
- Comparison table with at least 5 model variants
- Short report documenting methodology and results

🌟 **Excellence Indicators**:
- Multiple prompt designs tested and compared
- Training curves showing convergence
- Detailed per-class analysis (precision/recall)
- Computational efficiency analysis
- Clear explanation of tradeoffs between approaches

---

## Timeline Suggestion

| Phase | Estimated Time | Tasks |
|-------|---------------|-------|
| Setup | 2-3 hours | 1-4 |
| Agent 1 | 4-6 hours | 5-9 |
| Agent 2 | 8-10 hours | 10-16 |
| Agent 3 | 4-6 hours | 17-21 |
| Evaluation | 3-4 hours | 22-26 |
| Documentation | 3-4 hours | 27-30 |
| **Total** | **~30 hours** | |

---

## Questions to Consider During Development

1. **Agent 1**: Which prompt design works best? Why?
2. **Agent 2**: How much does fine-tuning improve over zero-shot?
3. **Agent 3**: How does discriminative classifier compare to generative?
4. **Overall**: What's the best accuracy vs. speed tradeoff?
5. **Practical**: Which agent would you deploy in production? Why?

---

## Bonus Presentation (Optional)

- **Date**: January 28th, 8:30 AM
- **Bonus**: +1 point
- **Deadline to book**: Email by January 20th
- Email: alberto.castellini@univr.it

---