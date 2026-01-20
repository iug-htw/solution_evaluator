# Investigating Bias — Reproducibility Guide (Multilingual Math Pipeline)

This repository contains an end-to-end pipeline for generating (or loading), translating, solving, and **pairwise-evaluating** math exercises across languages with multiple LLMs. It was built to make multilingual comparison reproducible and easy to extend to:
- new **solver models** (the models that generate solutions),
- new **judge models** (the models that rank solutions),
- new **languages** (beyond en/de/ar),
- or a new **exercise set** (as long as it is parallel across languages).

The core idea is simple: for each exercise, you generate one solution per language using a solver model, then have a held-out panel of judge LLMs rank those solutions (with randomized order to reduce position bias), and finally aggregate rankings via majority voting.

---

## Installation & Setup

Clone and install dependencies:

```bash
git clone https://github.com/iug-htw/solution_evaluator.git
cd solution_evaluator

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
````

### API Keys

This project calls commercial LLM APIs. Store keys **locally** in a `.env` file. Never hardcode keys and never commit them. The `.env` file is ignored by git.

Create `.env` in the repository root. Environment variables needed for the current setup:

```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
DASHSCOPE_API_KEY=your_alibaba_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

> Tip: if you add a new model/provider, add its key here and load it the same way in the client setup. Keep all secrets centralized in `.env`.

> NEVER hardcode keys or push them to the code repository.

---

## Prerequisites (Inputs the Pipeline Expects)

### 1) Exercises CSV per language (parallel datasets)

You need **one CSV file per language**, and they must be **equivalent across languages**:

* same number of rows
* same order (exercises should map to each other accross languages)
* same exercise IDs (if present)
* only the exercise text differs

Each CSV should include these columns:

* **Topic Area**
  High-level area (e.g., Numbers and Operations, Geometry, Probabilities)
* **Topic**
  Specific skill (e.g., Ordering numbers, Applying calculation strategies)
* **Progress Level**
  A–H mapped to grade levels (A–H ≈ Grades 1–10 under German K–10 framing)
* **Exercise**
  The actual task text in the target language

> Note: The original exercise set used in the study is saved as `math_exercises_en_en.csv`. It was translated to German and Arabic. This file is a good starting point for extending the experiment to more languages by translating it (see automated translation below) or by creating your own parallel set.

### 2) At least one solver model

A “solver model” is the LLM that generates step-by-step solutions for each exercise in each language.
You can use any model/provider, as long as you implement its API calls in the solver script (examples exist in the model-specific folders and the provided notebooks).

> Important: model versions get outdated quickly. If you are extending this work later, expect to swap in newer versions (or entirely new providers) and treat model selection as a configurable part of reproduction.

---

## Scripts (What Each Stage Does)

The pipeline is script-driven. Most scripts are “universal” and do not depend on a specific provider, except for solving (because each provider’s API differs).

### `_1_translate_tasks.py` [Universal]

**Purpose:** Translate the `Exercise` column from a source language (Default: English) into a target language using openAI API (default: GPT-4o-mini). \
**Location:** repository root \
**Entry point:** `translate_csv(input_csv, output_csv, target_language, source_language="English", model="gpt-4o-mini")` \

Behavior:

* loads the input exercises CSV
* sends each exercise through a translation prompt
* writes a new CSV in the target language

Notes:

* you can switch to any other source language via the `source_language` parameter
* you can switch to any other OpenAI model via the `model` parameter
* if you want to use a different provider for translation, adapt the script accordingly. 

---

### `_2_solve_tasks_<provider>.py` [Model-specific]

**Purpose:** Generate step-by-step solutions using a solver model. \
**Location:** inside each model directory (e.g., `./gpt_4o_mini/`, `./gemini/`, `./qwen/`) \
**Entry point:** \
`solve_tasks(input_file, output_file, model, prompt_prefix="Explain to me how I can solve this task", tasks_indices=None)`

Why model-specific?

each provider has different client setup, authentication, and calling conventions
so solving is implemented per model as separate scripts

Behavior:

* reads the input CSV row by row
* for each selected exercise, calls the model API with a prompt of the form
  `"<prompt_prefix>: <exercise>"`
* retries failed calls (e.g., rate limits) with backoff
* appends a new column like `"<model> solution"` to the output CSV

Notes:

* **prompt_prefix must be in the target language** for multilingual solving
  (e.g., German/Arabic/French prompt prefix, etc.)
* `tasks_indices` is useful for debugging, batching, partial reruns, or resuming. To solve all exercises leave this as `None`

---

### `_3_technical_terms.py` [Universal]

**Purpose:** Extract essential math technical terms per exercise, in the target language. \
**Location:** repository root \
**Entry point:**
`extract_technical_terms(input_file, output_file, target_language='en', model="gpt-4o-mini")` \

Why this matters:

* Technical terms are used during evaluation to help judges check whether key concepts/terminology were covered appropriately
* Extraction is grade-aware and avoids trivial terms

Behavior:

* Loads exercises
* Constructs language-specific prompts tailored to the grade/progress level
* Outputs a CSV with an added `Technical Terms` column (list-like content)

Notes:

* Current supported language codes in the script are `en`, `de`, `ar`
* To add a new language, extend the prompt templates / language handling in the helper function accordingly

---

### `_4_pairwise_evaluation.py` [Universal]

**Purpose:** Run pairwise (comparative) evaluation over multilingual solutions using a panel of LLM judges. \
**Location:** repository root \
**Entry point:**
`evaluate_explanations(files, technical_terms_files, current_model="gpt-4o-mini", output_dir="", tasks_indices=None)` \

What it does:

* loads solutions for each language (generated earlier)
* loads extracted technical terms per language (generated earlier)
* Maps the triplet (or N-tuple) of solutions per exercise, one for each language
* **randomizes the solution (languages) order per exercise** to mitigate position/order bias
* sends the triplet (or N-tuple) of solutions to judge models
* asks judges to rank solutions from best → worst and provide short justifications
* applies a held-out judging strategy: the solver model under evaluation is excluded from judges
  (Claude 3.5 Haiku is used as a neutral replacement in the current setup)
* writes/appends results to `judge_pairwise_evaluation.csv`

Inputs:

* `files`: mapping from language code/label → solutions CSV
  Example:

  ```python
  files = {
    "en": "path/to/solutions_en.csv",
    "de": "path/to/solutions_de.csv",
    "ar": "path/to/solutions_ar.csv"
  }
  ```

  To extend to more languages, add more entries:

  ```python
  files["fr"] = "path/to/solutions_fr.csv"
  ```

* `technical_terms_files`: mapping from language → technical terms CSV
  Each must include a `Technical Terms` column and align row-by-row with the exercises.

Extending to new models/providers:

* Update the helper `get_llm_client(model_name)` in this script to support the new judge model API client
* Then include the model name in the judge pool configuration
* In case the languages compared are more than 3: update the prompt used in the helper function `def rank_solutions(...)` to accomodate a bigger pool of solutions.

> Tip: It is recommended to keep the comparison pool small (e.g., up to three solutions or languages per evaluation). This reduces cognitive load on the judge LLMs and leads to more stable and reliable rankings.

---

### `_5_pairwise_results_clean.py` [Universal]

**Purpose:** Clean raw pairwise evaluation results and aggregate rankings via majority voting. \
**Location:** repository root \
**Entry point:** `pairwise_results_clean(file_dir="", solving_model="")` \

Behavior:

* Loads `judge_pairwise_evaluation.csv`
* Parses string rankings into structured dicts per judge
* Majority-votes “Best/Mid/Worst” (explicitly labels ties as `TIE`)
* Writes a cleaned results file: `pairwise_results_cleaned.csv`
* Generates a heatmap of ranking distributions by language

Extending to more judges:

* Update the `JUDGES` array at the top of the script
* Languages are parsed dynamically from the input, so you do not need to hardcode language lists

---

### `_6_justifications_analysis.py` [Universal]

**Purpose:** Analyze judge justifications (text) to understand why solutions were preferred or penalized. \
**Location:** repository root \
**Entry point:** `analyze_justifications_per_language(file_dir="", target_language='en')` \

Analyses included:

* n-gram frequency (uni/bi/tri-grams)
* sentiment distribution using `cardiffnlp/twitter-roberta-base-sentiment`
* topic modeling via LDA
* an optional GPT-based summary that synthesizes findings into research-style conclusions

Outputs:

* printed tables for n-grams
* sentiment distribution + plot
* topic clusters
* generated summary (if enabled)

Notes:

* This script is designed for qualitative insight and paper-style reporting
* It assumes justifications contain explicit references to language labels

---

## Running the Full Pipeline (End-to-End)

To reproduce the experiment for any set of models/languages, you run the scripts in order:

* prepare parallel exercise CSVs (one per language)
* translate (optional if you already have them)
* solve tasks per model per language
* extract technical terms per language
* pairwise evaluate per solver model
* clean/aggregate results (and optionally analyze justifications)

A fully worked, step-by-step execution example is included in the notebooks:

* `test_and_eval_pipeline-gpt.ipynb` (full GPT pipeline)
* `test_and_eval_pipeline-gemini.ipynb` (full Gemini pipeline)
* `test_and_eval_pipeline-qwen.ipynb` (full Qwen pipeline)

These notebooks are the best reference if you want to see the exact orchestration logic and file naming conventions used in practice.

---

## Extending the Pipeline (Languages and Models)

### Adding a new language

You’ll typically do the following:

* Produce a new exercises CSV in that language (recommended: translate the English base file to preserve parallelism)
* Solve tasks in that language for your solver model(s)
* Extract technical terms in that language (extend `_3_technical_terms.py` prompt templates if needed)
* Include the language in the `files` + `technical_terms_files` dicts for evaluation

### Adding a new solver model

A solver model requires:

* a model-specific `_2_solve_tasks_<provider>.py` script (copy an existing one and replace the client logic)
* an API key stored in `.env`
* updated notebook (or run commands) to generate solutions CSVs for each language

### Adding a new judge model

A judge model requires:

* adding client support in `_4_pairwise_evaluation.py` (usually inside `get_llm_client(model_name)`)
* adding its name to the judge pool configuration
* ensuring the model can follow the ranking + justification prompt format

### Avoiding self-judging

The evaluation script is designed so that:

* the solver model being evaluated is excluded from the judge panel
* a neutral model can replace it (Claude is used in the current setup)

Keep this pattern when adding new models to maintain impartiality.

---

## Reproducibility Notes

LLM outputs are stochastic. Exact counts may vary between runs due to:

* nondeterministic decoding
* model version updates by providers
* API-side changes
* transient formatting differences

To improve reproducibility across reruns:

* keep prompts fixed
* keep temperature and decoding parameters fixed (where supported)
* log model versions and timestamps
* save all raw outputs (`solutions/`, `judge_pairwise_evaluation.csv`) and rerun aggregation locally without re-calling APIs

---

## Where to Start

If you want the fastest path:

* open the notebook for the solver model closest to what you want (GPT/Gemini/Qwen)
* follow its steps to generate solutions, terms, and evaluation outputs
* then copy/adapt the same steps for your new language or new model

---

## Security Reminder

* never commit `.env`
* never print full keys to logs
* rotate keys if you accidentally expose them
