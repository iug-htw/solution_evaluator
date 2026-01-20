<p align="center">
  <img src="assets/htw_logo.jpg" alt="HTW Berlin Logo" width="300"/>
</p>

# Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs

_(Edu4AI Workshop @ ECAI 2025)_

This repository contains the codebase, datasets, and analysis scripts accompanying the paper:

**"Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs"**
Presented at: <br/>
_Edu4AI 2025: 2nd Workshop on Education for Artificial Intelligence,_
<br/> _co-located with the 28th European Conference on Artificial Intelligence (ECAI 2025),_
<br/> _Bologna, Italy_.

---

## Project Overview

Large Language Models (LLMs) are increasingly used to support students and teachers in education. However, most commercial models are trained on overwhelmingly English-centric data, raising concerns about **language-based disparities in clarity, accuracy, and pedagogical quality**.

This project introduces a **scalable, automated pipeline** to systematically investigate these disparities. Using 628 curriculum-aligned math exercises (Grades 2–10, German K–10 curriculum), we translate, solve, and evaluate problems across **English, German, and Arabic**.

Three LLMs (_**GPT-4o-mini**, **Gemini-2.5-Flash**, and **Qwen-Plus**_) produced solutions, which were then evaluated by a panel of LLM judges including **Claude 3.5 Haiku**.

Our results highlight **persistent linguistic bias** in educational AI, underscoring the need for more inclusive training and evaluation practices.

---

## Research Team

This project was conducted at **HTW Berlin – Hochschule für Technik und Wirtschaft Berlin** within the **KIWI Project**.

- [**Mariam Mahran**](https://mariamkhmahran.github.io/connect/) – Research Assistant, AI & Interpretability - HTW Berlin
- [**Prof. Dr. Katharina Simbeck**](https://iug.htw-berlin.de) – Professor of Business Informatics (Information Management) - HTW Berlin

---

## Pipeline Overview

The multilingual pipeline is organized into four major stages:

### 1. Dataset Preparation

- **Problem generation**: 628 math exercises were created in English, aligned with the German K–10 curriculum.
  _Exercises were generated using ChatGPT-4o in a manual, iterative process guided by curriculum-aligned topic names and learning objectives._
- **Translation**: Exercises were translated into German and Arabic.
- **Technical terms extraction**: Key mathematical terms were identified for each exercise to support later evaluation.

### 2. Solution Generation

- Three commercial LLMs — **GPT-4o-mini**, **Gemini-2.5-Flash**, and **Qwen-Plus** — produced step-by-step solutions for all exercises in English, German, and Arabic.

Note that each model uses its own dedicated solver script, since APIs differ in setup and calling conventions:

- **GPT-4o-mini**: `gpt_4o_mini/_2_solve_tasks_gpt.py`
- **Gemini-2.5-Flash**: `gemini/_2_solve_tasks_gemini.py`
- **Qwen-Plus**: `qwen_plus/_2_solve_tasks_qwen.py`

Each script loads the corresponding API key, constructs prompts in the target language, and writes the model’s step-by-step solutions to CSV files.

### 3. Evaluation

- **Judging framework**: Solutions were assessed by a panel of LLM judges.
- **Held-out strategy**: The model under evaluation was excluded from the judging panel and replaced with a neutral model (**Claude 3.5 Haiku**).
- **Randomized presentation**: Solutions were shown in shuffled order to reduce position bias.
- **Comparative assessment**: Judges ranked the three solutions (1 = best, 3 = worst) and provided short justifications.
- **Majority voting**: Rankings were aggregated across judges to determine the best-performing language per exercise.
  _If no consensus was reached, the result was labeled as a tie (TIE)._

### 4. Justifications Analysis

- N-gram, sentiment, and topic modeling of judge justifications.

---

## Repository Structure

```
SOLUTION_EVALUATOR/
│
├── archive/                     # Old experiments, preliminary query sets, early analyses
├── assets/                      # Static assets for the repository
│
├── gpt_4o_mini/                 # Outputs and results for GPT-4o-mini
│   ├── _2_solve_tasks_gpt.py    # Solve tasks with GPT-4o-mini
│   └── (other model-specific output files...)
│
├── gemini/                      # Outputs and results for Gemini-2.5-Flash
│   ├── _2_solve_tasks_gemini.py    # Solve tasks with Gemini-2.5-Flash
│   └── (other model-specific output files...)
│
├── qwen_plus/                   # Outputs and results for Qwen-Plus
│   ├── _2_solve_tasks_qwen.py       # Solve tasks with Qwen-Plus
│   └── (other model-specific output files...)
│
├── _1_translate_tasks.py        # Translate exercises into target language
├── _3_technical_terms.py        # Extract key technical terms per exercise
├── _4_pairwise_evaluation.py    # Pairwise ranking evaluation with justifications
├── _5_pairwise_results_clean.py # Majority-vote cleaning & heatmap visualization
├── _6_justifications_analysis.py   # Justification text analysis (ngrams, sentiment, topics)
│
├── justifications_analysis.ipynb         # Notebook for justification analysis
├── test_and_eval_pipeline-gpt.ipynb      # Full GPT pipeline
├── test_and_eval_pipeline-gemini.ipynb   # Full Gemini pipeline
├── test_and_eval_pipeline-qwen.ipynb     # Full Qwen pipeline
│
├── math_exercises_en_en.csv     # Original English dataset
├── math_exercises_en_de.csv     # Translated German dataset
├── math_exercises_en_ar.csv     # Translated Arabic dataset
├── technical_terms_en.csv       # Extracted terms (English)
├── technical_terms_de.csv       # Extracted terms (German)
├── technical_terms_ar.csv       # Extracted terms (Arabic)
│
├── Part_C_Mathematics_2015_11_10.pdf  # Reference German curriculum
├── requirements.txt                   # Minimal dependencies
├── README.md                    # Project overview (this file)
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/iug-htw/solution_evaluator.git
cd solution_evaluator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### API Keys

This project requires access to:

- **OpenAI API Key** (for GPT-4o-mini and as a judge)
- **Google Gemini API Key** (for Gemini-2.5-Flash and as a judge)
- **Alibaba Cloud API Key** (for Qwen-Plus and as a judge)
- **Anthropic API Key** (for Claude 3.5 Haiku as a judge)

### Example `.env` File

```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
DASHSCOPE_API_KEY=your_alibaba_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

---

## Reproducing the Experiment

This repository includes a detailed, step-by-step guide for reproducing the full multilingual evaluation pipeline, including translation, solution generation, LLM-based judging, and result aggregation.

➡️ See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full instructions.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{MahranSimbeck2025Edu4AI,
  author    = {Mariam Mahran and Katharina Simbeck},
  title     = {Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs},
  booktitle = {Proceedings of the 2nd International Workshop on Education for Artificial Intelligence (edu4AI 2025), ECAI},
  series    = {CEUR Workshop Proceedings},
  volume    = {4114},
  year      = {2025},
  address   = {Bologna, Italy},
  url       = {https://ceur-ws.org/Vol-4114/6_paper.pdf},
  issn      = {1613-0073}
}
```
}

---

## Acknowledgments

This work was carried out as part of the **KIWI Project**, generously funded by the **Federal Ministry of Education and Research (BMBF)**.

We gratefully acknowledge their support, which enabled this research.

---

<p align="center"><i>Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs - HTW Berlin, 2025</i></p>