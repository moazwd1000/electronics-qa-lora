# electronics-qa-lora


# Fine-Tuning Qwen2.5-1.5B for Electronics Question Answering with LoRA

A small-scale supervised fine-tuning (SFT) project that adapts **Qwen2.5-1.5B-Instruct** to provide undergraduate-level explanations for fundamental electronics concepts.

The project uses **LoRA (Low-Rank Adaptation)** with Hugging Face **PEFT** and **TRL**, demonstrating parameter-efficient fine-tuning of a pretrained language model.

---

## Project Overview

The goal was to fine-tune a pretrained instruction-following language model for one narrow task:

> **Answering undergraduate electronics questions clearly and technically.**

Instead of training a generic chatbot, the dataset focuses on five electronics topics:

* Kirchhoff's Laws
* Diodes
* BJTs
* MOSFETs
* Operational Amplifiers (Op-Amps)

The project was designed as a practical experiment to understand the complete LoRA/SFT workflow, including conversational formatting, chat templates, loss masking, parameter-efficient training, and evaluation.

---

## Model

**Base model:** Qwen2.5-1.5B-Instruct

The base model contains approximately **1.54 billion parameters**.

For fine-tuning, LoRA was applied to selected attention modules rather than updating the entire model.

### LoRA configuration

* LoRA rank: `8`
* LoRA alpha: `16`
* Target modules: `q_proj`, `v_proj`
* Trainable parameters: **1,089,536**
* Total parameters: **1,544,803,840**
* Trainable percentage: **0.0705%**

This means that less than 0.1% of the model's parameters were updated during training.

---

## Dataset

The dataset contains **250 question-answer examples** covering five electronics topics.

| Topic            | Examples |
| ---------------- | -------: |
| Kirchhoff's Laws |       51 |
| Diodes           |       50 |
| BJTs             |       50 |
| MOSFETs          |       50 |
| Op-Amps          |       49 |
| **Total**        |  **250** |

### Dataset split

| Split      | Examples | Purpose                                     |
| ---------- | -------: | ------------------------------------------- |
| Training   |      200 | Used to train the LoRA adapter              |
| Validation |       25 | Used to monitor performance during training |
| Test       |       25 | Used only for final evaluation              |

No test examples were used during training.

---

## Dataset Format

Each example contains:

* `id`
* `topic`
* `difficulty`
* `instruction`
* `question`
* `answer`

The examples were converted into a conversational format compatible with the Qwen chat template:

```text
System → instruction
User → question
Assistant → answer
```

For example:

```text
System:
Explain this concept as if you were tutoring an undergraduate engineering student.

User:
Why does a diode conduct current in only one direction?

Assistant:
The PN junction's depletion region and built-in electric field...
```

---

## Training Approach

The project uses **Supervised Fine-Tuning (SFT)** with LoRA.

The base model is frozen, while the small LoRA adapter parameters are trained.

### Training configuration

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

### Assistant-only loss

The conversational dataset contains system, user, and assistant messages.

During training, the loss was calculated only on the assistant response.

Conceptually:

```text
System message      → ignored
User question       → ignored
Assistant response  → loss calculated
```

This prevents the model from being trained to reproduce the instruction and question and focuses the optimization on generating the desired answer.

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

The chat template was also used during inference to ensure that prompts have the same conversational structure as the training examples.

---

## Training Results

The model was trained for three epochs.

| Epoch | Training Loss | Validation Loss | Token Accuracy |
| ----: | ------------: | --------------: | -------------: |
|     1 |        1.7235 |          1.6721 |         57.66% |
|     2 |        1.5446 |          1.5818 |         59.87% |
|     3 |        1.4781 |          1.5563 |         60.94% |

Both training and validation loss decreased throughout training.

This indicates that the LoRA adapter was learning the task without obvious validation-loss divergence during the three training epochs.

---

## Test Evaluation

The final evaluation was performed on the **25 examples that were not used during training or validation**.

### Test loss

| Model                      |  Test Loss |
| -------------------------- | ---------: |
| Base Qwen2.5-1.5B-Instruct | **2.1132** |
| Qwen2.5-1.5B + LoRA        | **1.6593** |

### Result

The fine-tuned model reduced test loss by:

**0.4539**

or approximately:

**21.5%**

compared with the original base model.

This provides evidence that the LoRA adapter improved the model's ability to predict the target electronics answers on unseen examples.

---

## Example

### Question

> Why does a diode conduct current in only one direction?

### Reference Answer

> The PN junction's depletion region and built-in electric field oppose current flow in reverse bias by blocking majority carrier movement, while in forward bias the applied voltage overcomes this barrier and allows majority carriers to flow across the junction.

The project also compares the behavior of the original Qwen model and the LoRA fine-tuned model on unseen electronics questions.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face PEFT
* Hugging Face TRL
* Qwen2.5-1.5B-Instruct
* LoRA
* Google Colab
* NVIDIA Tesla T4

---

## Project Structure

```text
electronics-qa-lora/
│
├── data/
│   ├── train.json
│   ├── valid.json
│   └── test.json
│
├── notebooks/
│   └── electronics_qa_lora.ipynb
│
├── adapter/
│   └── ...
│
├── README.md
└── requirements.txt
```

The exact structure may vary depending on how the training notebook and adapter files are organized.

---

## What I Learned

This project was primarily built to understand the mechanics of parameter-efficient fine-tuning rather than simply using a pretrained model.

Key concepts explored:

* Full fine-tuning vs. frozen models
* Parameter-efficient fine-tuning
* LoRA and low-rank adaptation
* LoRA rank and alpha
* Selecting target modules
* Chat templates
* Conversational datasets
* Instruction formatting
* Assistant-only loss
* Supervised fine-tuning
* Training and validation loss
* Test-set evaluation
* Comparing a base model with a fine-tuned model

---

## Limitations

This is a small experimental project and the results should be interpreted accordingly.

* The dataset contains only 250 examples.
* Only five electronics topics are covered.
* The test set contains only 25 examples.
* The dataset is relatively homogeneous in writing style.
* Test loss does not fully measure the quality of natural-language explanations.
* The model can still produce technically irrelevant or incomplete answers.
* A larger and more diverse dataset would be required for stronger conclusions.

The goal of this project was therefore **learning and demonstrating the LoRA/SFT workflow**, rather than producing a production-ready electronics tutor.

---

## Future Improvements

Possible extensions include:

* Increase the size and diversity of the electronics dataset.
* Add numerical circuit-analysis questions.
* Include circuit diagrams and multimodal data.
* Add more difficulty levels.
* Compare different LoRA ranks.
* Experiment with different target modules.
* Compare different learning rates and training epochs.
* Perform a larger human evaluation.
* Add automated evaluation alongside test loss.
* Deploy the fine-tuned adapter as an interactive electronics tutor.

---

## Conclusion

This project demonstrates a complete parameter-efficient fine-tuning workflow for a small instruction-following language model.

Using only **0.0705% of the model's parameters**, LoRA fine-tuning reduced test loss from **2.1132 to 1.6593**, a **21.5% reduction**, on a held-out electronics question-answering dataset.

The project demonstrates how a pretrained LLM can be adapted to a narrow technical domain without updating the full model.
:::
