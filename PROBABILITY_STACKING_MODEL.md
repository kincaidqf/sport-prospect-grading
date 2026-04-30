# PROBABILITY STACKING MODEL

## Objective
Build a multi-modal model that combines structured college statistics and unstructured scouting reports to predict NBA prospect outcomes:

**Classes:** Bust / Bench / Starter / Star

This system will use a **probability stacking (meta-model) approach** to learn how to optimally combine signals from different model types.

---

## High-Level Architecture

```
stats --------> Model A (best tabular model) ───┐
stats --------> Model B (best tabular model) ───┼──→ Meta-Model → Final Prediction
text ---------> DistilBERT ---------------------┘
```

Each base model produces **class probabilities**, which are then used as inputs to a second-stage model (meta-model).

---

## Base Models (Stage 1)

### Important Note
- Logistic Regression and XGBoost are **initial placeholders**
- These should be replaced with the **best-performing models** identified during experimentation
- The architecture is **model-agnostic**

### Expected Inputs/Outputs

Each base model outputs:

```
[p_bust, p_bench, p_starter, p_star]
```

Models:

- **Model A (Tabular)**  
  Example: Logistic Regression (placeholder)

- **Model B (Tabular)**  
  Example: XGBoost (placeholder)

- **Text Model**  
  DistilBERT fine-tuned on scouting reports

---

## Meta-Model (Stage 2)

### Input Features

Concatenate probabilities from all base models:

```
[
  modelA_bust, modelA_bench, modelA_starter, modelA_star,
  modelB_bust, modelB_bench, modelB_starter, modelB_star,
  bert_bust,  bert_bench,  bert_starter,  bert_star
]
```

Optional additions:

- Confidence metrics:
  - Max probability
  - Entropy
  - Margin between top two classes
- Metadata:
  - Class year
  - Age
  - Minutes played
  - Team difficulty

### Output

```
Final prediction → Bust / Bench / Starter / Star
```

---

## Why This Works

The meta-model learns:

- When to trust **stats vs text**
- How to handle **model disagreement**
- How to leverage **confidence levels**
- Cross-signal patterns (e.g., strong stats + weak scouting report)

---

## Training Procedure (Critical)

Use **out-of-fold (OOF) predictions**:

1. Split training data into folds
2. Train base models on K-1 folds
3. Generate predictions on the held-out fold
4. Repeat for all folds
5. Train meta-model on these OOF predictions

**Do NOT train the meta-model on predictions from models trained on the same data** (prevents overfitting)

---

## Implementation Steps

### Step 1 — Train Base Models
- Train tabular models on stats
- Fine-tune DistilBERT on scouting reports

### Step 2 — Generate Probabilities
- Output class probabilities for all models

### Step 3 — Build Meta Dataset
- Concatenate probabilities into feature vectors

### Step 4 — Train Meta-Model
- Start with Logistic Regression
- Compare against XGBoost / small neural net

### Step 5 — Evaluate
- Compare against individual base models
- Validate improvement in classification accuracy and calibration

---

## Future Extensions

- Replace placeholders with best-performing models
- Add embedding-level fusion (DistilBERT hidden states)
- Implement gated fusion (learn dynamic weighting)
- Explore regression targets (e.g., career value metrics)

---

## Summary

This project’s final goal is a **modular, model-agnostic multi-modal system** where:

- Any strong tabular model can plug in
- Any strong text model can plug in
- A meta-model learns how to combine them optimally

The probability stacking framework ensures:
- Flexibility
- Interpretability
- Strong baseline performance
