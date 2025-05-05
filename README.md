# Multilingual Evaluation of LLM-Generated Educational Math Content

This repository presents a multilingual evaluation pipeline for assessing how large language models (LLMs) generate and explain educational math content across languages. The project investigates whether linguistic differences affect the pedagogical quality of LLM outputs in math instruction aligned with the German K-10 curriculum.

## 🧠 Project Overview

We introduce a comprehensive pipeline that:

1. **Generates math exercises** using GPT-4o aligned with the German school curriculum.
2. **Translates** these exercises into German and Arabic.
3. **Solves** each version (English, German, Arabic) using three LLMs:
   - GPT-4o-mini
   - Gemini 1.5 Flash
   - Qwen-Plus
4. **Evaluates** the pedagogical quality of each solution using LLM-based judges across key metrics:
   - Comprehension
   - Clarity
   - Accuracy
   - Use of mathematical terminology
   - Learning appropriateness, and more.

## 🔍 Key Findings

- **English** explanations consistently received the highest ratings.
- **German** explanations were rated moderately.
- **Arabic** explanations were rated lowest, despite its global prominence.

These results highlight **linguistic performance disparities** in LLMs, raising concerns about equitable AI in educational contexts.

## 📂 Repository Structure

Each of the three main jupyter notebooks correspond to a different LLM model:

- `test_and_eval_pipeline.ipynb` -> `gpt-4o-mini`
- `test_and_eval_pipeline-gemini.ipynb` -> `gemini-1.5-flash`
- `test_and_eval_pipeline-qwen.ipynb` -> `qwen-plus`

All notebooks follow the same pipeline logic.

## 🛠️ Pipeline Steps

The pipeline is modular, using a series of helper scripts:

| Module                                        | Description                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------- |
| `_0_0_generate_tasks.py`                      | Generates exercises using gpt-4o                                                  |
| `_0_prepare_tasks.py`                         | Cleans and formats the original list of math exercises                            |
| `_1_translate_tasks.py`                       | Translates exercises to German and Arabic                                         |
| `_2_solve_tasks.py`                           | Uses LLMs to solve problems in each language                                      |
| `_3_technical_terms.py`                       | Extracts domain-specific mathematical terms                                       |
| `_4_evaluate_solution.py`                     | Judges each solution using pedagogical criteria (direct scoring, non-comparative) |
| `_5_cross_lang_validation.py`                 | Compares scores across languages                                                  |
| `_6_find_differently_performing_exercises.py` | Highlights exercises with large score differentials                               |
| `_7_pairwise_evaluation.py`                   | Performs pairwise LLM evaluation of explanations                                  |
| `_8_pairwise_results_clean.py`                | Cleans and aggregates pairwise results                                            |

## 📊 Outputs

- `*.csv` files containing cleaned exercises, translations, LLM solutions, technical terms, and evaluation scores.
- Visualizations comparing performance across:
  - Languages
  - Evaluation metrics
  - LLM models

## 📊 Outputs Directories

| Directory    | Content                                                                           |
| ------------ | --------------------------------------------------------------------------------- |
| `/`          | Contains main scripts, shared outputs, and output files specific to `gpt-4o-mini` |
| `/gemini`    | Contains gemini-custom scripts and output files specific to `gemini-1.5-flash`    |
| `/qwen_plus` | Contains gemini-custom scripts and output files specific to `qwen-plus`           |

## 📎 Dependencies

- Python 3.8+
- `pandas`, `seaborn`, `matplotlib`, `numpy`
- API keys for OpenAI (or access credentials for other LLMs)

Install dependencies:

```bash
pip install -r requirements.txt
```
