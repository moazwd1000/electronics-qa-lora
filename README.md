````markdown
# electronics-qa-lora

# Fine-Tuning Qwen2.5-1.5B for Electronics Question Answering with LoRA and QLoRA

A small-scale supervised fine-tuning (SFT) project that adapts **Qwen2.5-1.5B-Instruct** to provide undergraduate-level explanations for fundamental electronics concepts.

The project explores two parameter-efficient fine-tuning approaches:

- **LoRA (Low-Rank Adaptation)**
- **QLoRA (Quantized Low-Rank Adaptation)**

Both approaches use Hugging Face **PEFT** and **TRL** and are evaluated on the same held-out electronics question-answering test set.

The project focuses on understanding the complete fine-tuning workflow rather than building a production-ready electronics tutor.

---

## Project Overview

The goal was to fine-tune a pretrained instruction-following language model for one narrow task:

> **Answering undergraduate electronics questions clearly and technically.**

Instead of training a general-purpose chatbot, the dataset focuses on five fundamental electronics topics:

- Kirchhoff's Laws
- Diodes
- BJTs
- MOSFETs
- Operational Amplifiers (Op-Amps)

The project was designed as a practical experiment to understand:

- Supervised Fine-Tuning (SFT)
- Parameter-efficient fine-tuning
- LoRA
- QLoRA
- Low-rank adapters
- 4-bit quantization
- Chat templates
- Conversational datasets
- Assistant-only loss masking
- Training and validation monitoring
- Held-out test evaluation
- Perplexity
- LoRA vs. QLoRA trade-offs

---

## Model

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`

The model contains approximately **1.54 billion parameters**.

Instead of updating the entire model, the project uses LoRA adapters to update a very small subset of parameters.

---

## Dataset

The dataset contains **250 question-answer examples** covering five electronics topics.

| Topic | Examples |
|---|---:|
| Kirchhoff's Laws | 51 |
| Diodes | 50 |
| BJTs | 50 |
| MOSFETs | 50 |
| Op-Amps | 49 |
| **Total** | **250** |

### Dataset Split

| Split | Examples | Purpose |
|---|---:|---|
| Training | 200 | Used to train the adapters |
| Validation | 25 | Used to monitor performance during training |
| Test | 25 | Used only for final evaluation |

The test set was kept separate from training and validation and was used for the final comparison between the base model and fine-tuned models.

---

## Dataset Format

Each example contains:

- `id`
- `topic`
- `difficulty`
- `instruction`
- `question`
- `answer`

The examples were converted into a conversational format compatible with the Qwen chat template:

```text
System → instruction
User → question
Assistant → answer
````

For example:

```text
System:
Explain this concept as if you were tutoring an undergraduate engineering student.

User:
Why does a diode conduct current in only one direction?

Assistant:
The PN junction's depletion region and built-in electric field...
```

The Qwen chat template was used during training and inference to maintain consistent conversational formatting.

---

## Training Approach

The project implements two related approaches using the same base model, dataset, and core LoRA configuration.

```text
                    Qwen2.5-1.5B-Instruct
                              │
                ┌─────────────┴─────────────┐
                │                           │
              LoRA                        QLoRA
                │                           │
        Full-precision base        4-bit quantized base
                │                           │
        LoRA adapters                LoRA adapters
                │                           │
                └─────────────┬─────────────┘
                              │
                         SFT Training
                              │
                    Held-out Test Set
```

---

## LoRA

LoRA freezes the pretrained model weights and introduces trainable low-rank matrices into selected model layers.

### LoRA Configuration

| Parameter            |              Value |
| -------------------- | -----------------: |
| LoRA rank (`r`)      |                  8 |
| LoRA alpha           |                 16 |
| LoRA dropout         |               0.05 |
| Target modules       | `q_proj`, `v_proj` |
| Task type            |        `CAUSAL_LM` |
| Trainable parameters |          1,089,536 |
| Total parameters     |      1,544,803,840 |
| Trainable percentage |        **0.0705%** |

Only approximately **0.07% of the model parameters** were trainable.

---

## QLoRA

QLoRA extends the LoRA approach by loading the base model using **4-bit quantization** while keeping the LoRA adapters trainable.

The project uses:

```text
4-bit quantization
        ↓
NF4 quantization type
        ↓
Double quantization
        ↓
BF16 computation
        ↓
LoRA adapters
        ↓
SFT training
```

### QLoRA Configuration

| Parameter           |              Value |
| ------------------- | -----------------: |
| Quantization        |              4-bit |
| Quantization type   |                NF4 |
| Double quantization |            Enabled |
| Compute dtype       |               BF16 |
| LoRA rank           |                  8 |
| LoRA alpha          |                 16 |
| LoRA dropout        |               0.05 |
| Target modules      | `q_proj`, `v_proj` |
| Task type           |        `CAUSAL_LM` |

The same LoRA architecture was used for both LoRA and QLoRA so that the comparison primarily focuses on the effect of quantizing the base model.

---

## Training Configuration

The same core SFT configuration was used for both LoRA and QLoRA.

| Parameter               |           Value |
| ----------------------- | --------------: |
| Epochs                  |               3 |
| Per-device batch size   |               2 |
| Gradient accumulation   |               4 |
| Effective batch size    |               8 |
| Learning rate           |          `2e-4` |
| Maximum sequence length |             512 |
| Evaluation              |     Every epoch |
| Saving                  |     Every epoch |
| Loss                    |  Assistant-only |
| GPU                     | NVIDIA Tesla T4 |

---

## Assistant-Only Loss

The conversational dataset contains:

```text
System message
User question
Assistant answer
```

The training objective focuses the loss only on the **assistant response**.

Conceptually:

```text
System message      → ignored
User question       → ignored
Assistant response  → loss calculated
```

This focuses the optimization on generating the desired answer rather than reproducing the system instruction and user question.

The SFT configuration uses:

```python
assistant_only_loss=True
```

---

## Chat Template

The Qwen chat template was used to convert conversational messages into the format expected by the model.

The resulting structure follows the Qwen message format:

```text
<|im_start|>system
...
<|im_end|>

<|im_start|>user
...
<|im_end|>

<|im_start|>assistant
...
<|im_end|>
```

The same conversational structure was used during inference to maintain consistency with the training format.

---

# Training Results

## LoRA

The LoRA model was trained for three epochs.

| Epoch | Training Loss | Validation Loss | Token Accuracy |
| ----: | ------------: | --------------: | -------------: |
|     1 |        1.7235 |          1.6721 |         57.66% |
|     2 |        1.5446 |          1.5818 |         59.87% |
|     3 |        1.4781 |          1.5563 |         60.94% |

Both training and validation loss decreased throughout training, indicating that the adapter was learning the target task without obvious validation-loss divergence during the three epochs.

---

## QLoRA

The QLoRA model was also trained for three epochs.

| Epoch | Training Loss | Validation Loss | Mean Token Accuracy |
| ----: | ------------: | --------------: | ------------------: |
|     1 |        1.7960 |          1.7196 |              55.97% |
|     2 |        1.6034 |          1.6189 |              58.56% |
|     3 |        1.5362 |          1.5931 |              58.74% |

The QLoRA training run completed **75 optimization steps over three epochs**.

The validation loss decreased from **1.7196 to 1.5931**, indicating that the QLoRA adapter learned the target task during training.

---

# Test Evaluation

The final evaluation was performed on the **25 held-out test examples**.

The evaluation loss was calculated over the target answer tokens rather than the entire conversational prompt.

## Test Loss and Perplexity

| Model                      |  Test Loss | Perplexity |
| -------------------------- | ---------: | ---------: |
| Base Qwen2.5-1.5B-Instruct | **2.1132** |   **8.27** |
| Qwen2.5-1.5B + LoRA        | **1.6593** |   **5.26** |
| Qwen2.5-1.5B + QLoRA       | **1.7288** |   **5.63** |

Perplexity was calculated using:

[
PPL=e^{Loss}
]

Lower loss and lower perplexity indicate better prediction of the target answer tokens.

---

# LoRA vs. QLoRA

Both parameter-efficient fine-tuning approaches improved performance compared with the original base model.

### Base → LoRA

Test loss decreased:

```text
2.1132 → 1.6593
```

This represents approximately a:

**21.5% reduction in test loss.**

Perplexity decreased from approximately:

```text
8.27 → 5.26
```

### Base → QLoRA

Test loss decreased:

```text
2.1132 → 1.7288
```

This represents approximately an:

**18.2% reduction in test loss.**

Perplexity decreased from approximately:

```text
8.27 → 5.63
```

### Overall Comparison

| Model |  Test Loss | Perplexity |
| ----- | ---------: | ---------: |
| Base  |     2.1132 |       8.27 |
| LoRA  | **1.6593** |   **5.26** |
| QLoRA |     1.7288 |       5.63 |

For this experiment, **LoRA achieved the lowest test loss and perplexity**.

QLoRA performed slightly worse on these metrics, but it demonstrated that the model could be fine-tuned using a **4-bit quantized base model** while still achieving a substantial improvement over the original model.

Therefore, the results demonstrate a practical trade-off between **fine-tuning performance and memory-efficient quantized training**.

---

# Qualitative Evaluation

Numerical metrics such as loss and perplexity do not fully measure whether an electronics explanation is technically correct or useful.

The models were therefore also evaluated through direct generation on unseen electronics questions.

The evaluation compares:

```text
Question
   ↓
Base Model
   ↓
LoRA Model
   ↓
QLoRA Model
   ↓
Reference Answer
```

Generated answers can be evaluated based on:

* Technical correctness
* Relevance
* Completeness
* Clarity
* Ability to explain undergraduate electronics concepts

This qualitative evaluation is important because a lower token-level loss does not necessarily guarantee that every generated answer is technically correct.

---

# Evaluation Metrics

The project uses multiple evaluation perspectives:

### Training Loss

Measures how well the model fits the training examples.

### Validation Loss

Monitors performance on unseen validation examples during training.

### Test Loss

Measures performance on the final held-out test set.

### Perplexity

Calculated from test loss:

[
PPL=e^{Loss}
]

Lower perplexity indicates that the model assigns higher probability to the target answer tokens.

### Qualitative Generation

Generated answers are compared with reference answers to assess:

* Technical correctness
* Relevance
* Completeness
* Clarity
* Explanation quality

---

# Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face PEFT
* Hugging Face TRL
* BitsAndBytes
* Qwen2.5-1.5B-Instruct
* LoRA
* QLoRA
* Google Colab
* NVIDIA Tesla T4

---

# Project Structure

```text
electronics-qa-lora/
│
├── data.json
│
├
│── electronics_qa_lora.ipynb
│
├
│   
│
├── README.md

```

The exact structure may vary depending on how the notebook, datasets, and adapter checkpoints are organized.

---

# What I Learned

This project was built to understand the mechanics of parameter-efficient fine-tuning rather than simply using a pretrained model.

Key concepts explored:

* Full fine-tuning vs. parameter-efficient fine-tuning
* LoRA and low-rank adaptation
* QLoRA
* 4-bit model quantization
* NF4 quantization
* Double quantization
* LoRA rank and alpha
* Target module selection
* Frozen base model weights
* Trainable parameter analysis
* Chat templates
* Conversational datasets
* Instruction formatting
* Assistant-only loss
* Supervised fine-tuning
* Training and validation monitoring
* Held-out test evaluation
* Cross-entropy loss
* Perplexity
* Qualitative generation evaluation
* LoRA vs. QLoRA performance trade-offs

---

# Limitations

This is a small experimental project and the results should be interpreted accordingly.

* The dataset contains only **250 examples**.
* Only five electronics topics are covered.
* The test set contains only **25 examples**.
* The dataset is relatively homogeneous in writing style.
* The evaluation set is small, so differences in test loss should not be treated as statistically conclusive.
* Test loss and perplexity do not fully measure the quality of natural-language explanations.
* The model can still produce technically incorrect, irrelevant, or incomplete answers.
* LoRA and QLoRA were evaluated using a single primary configuration.
* No large-scale human evaluation was performed.
* GPU memory usage was not systematically benchmarked between LoRA and QLoRA in this experiment.

The goal of the project is therefore **to demonstrate and understand the LoRA/QLoRA SFT workflow**, rather than claim production-level performance.

---

# Future Improvements

Possible extensions include:

### Dataset

* Increase the dataset size and diversity.
* Add numerical circuit-analysis questions.
* Add multi-step problem-solving examples.
* Add more difficulty levels.
* Add questions requiring equations and derivations.
* Include circuit diagrams and multimodal data.

### Fine-Tuning

* Compare different LoRA ranks.
* Experiment with different target modules.
* Test different learning rates.
* Investigate different training schedules.
* Compare additional quantization configurations.
* Perform multiple training runs with different random seeds.

### Evaluation

* Increase the test-set size.
* Add automated answer-quality metrics.
* Perform human evaluation by electronics students or instructors.
* Evaluate individual electronics topics separately.
* Measure factual and technical correctness rather than relying only on token-level metrics.
* Compare generation quality across Base, LoRA, and QLoRA.

### Deployment

* Package the trained adapter.
* Build an interactive electronics tutoring interface.
* Deploy the model with a lightweight inference backend.
* Add retrieval-augmented generation for textbook/reference material.
* Explore multimodal electronics questions involving circuit diagrams.

---

# Conclusion

This project demonstrates a complete **parameter-efficient fine-tuning workflow** for adapting Qwen2.5-1.5B-Instruct to a narrow electronics question-answering domain.

Two approaches were investigated:

* **LoRA**
* **QLoRA**

Using only **1,089,536 trainable parameters**, approximately **0.0705% of the 1.54B-parameter model**, the LoRA experiment reduced held-out test loss from **2.1132 to 1.6593**, corresponding to a **21.5% reduction**.

The QLoRA experiment reduced test loss to **1.7288**, corresponding to an approximately **18.2% reduction** relative to the base model.

In this experiment, **LoRA achieved slightly better test performance than QLoRA** based on test loss and perplexity. However, QLoRA demonstrated that the model could be fine-tuned while using a **4-bit NF4 quantized base model**, providing a more memory-efficient training approach.

Overall, the project demonstrates how **parameter-efficient fine-tuning can adapt a pretrained language model to a narrow technical domain without updating the full model**, while providing a practical comparison between standard LoRA and quantized QLoRA fine-tuning.

```
```
