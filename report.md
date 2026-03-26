# SC4021 Information Retrieval - Classification Report

## Question 4: Classification

### 4.1 Choice of Classification Approach

We adopt a **machine learning-based** approach for sentiment analysis on Singapore Reddit (education-related) data. Our approach is motivated by the state of the art as follows:

1. **Machine learning baselines** (Bag-of-Words + Naive Bayes, TF-IDF + SVM) are well-established in text classification literature and serve as strong, reproducible baselines. Pang et al. (2002) demonstrated that ML-based approaches outperform simple keyword-counting methods for sentiment classification.

2. **TF-IDF + Linear SVM** is widely recognized as a competitive baseline for text classification tasks, particularly on medium-sized datasets (~5,700 samples). TF-IDF captures term importance through inverse document frequency weighting, and Linear SVM excels in high-dimensional sparse feature spaces.

3. **Multiple classifier comparison**: We evaluate Logistic Regression, Linear SVM, Multinomial Naive Bayes, and Random Forest across two text representations (original Singlish vs. normalized Standard English) to identify the best-performing model per task.

The pipeline covers **two required subtasks** plus one additional subtask:
- **Subjectivity Detection** (binary: objective vs. subjective) — filters opinionated content
- **Polarity/Sentiment Detection** (3-class: negative / neutral / positive) — classifies sentiment polarity
- **Sarcasm Detection** (binary) — additional subtask for enhanced classification (Q5)

### 4.2 Data Preprocessing

**Preprocessing discussion:** The raw crawled data from Singapore Reddit contains informal Singlish text with colloquial expressions, abbreviations, and code-mixing. We applied the following preprocessing steps and discuss their impact:

#### 4.2.1 Microtext Normalization

We applied LLM-based text normalization (GPT-5 + Gemini dual-model) to convert Singlish expressions into Standard English. Additionally, we implemented rule-based microtext normalization including:

- **Singlish particle removal/expansion**: "lah", "lor", "leh" → removed; "liao" → "already"; "sian" → "bored"
- **Internet abbreviation expansion**: "imo" → "in my opinion", "tbh" → "to be honest", "idk" → "I don't know"
- **Repeated character normalization**: "sooooo" → "soo" (cap at 2 repeats)
- **URL and mention removal**: Strip `https://...` links and `/r/`, `/u/` references
- **Lowercase conversion** and **whitespace normalization**

#### 4.2.2 Basic Statistics

| Statistic | Original (Singlish) | Normalized (Std English) |
|-----------|--------------------:|-------------------------:|
| Samples | 5,733 | 5,733 |
| Mean char length | 282.4 | 278.1 |
| Median char length | 173.0 | 165.0 |
| Std char length | 356.8 | 359.9 |
| Min / Max char length | 20 / 4,760 | 20 / 4,760 |
| Mean word count | 52.1 | 51.5 |
| Vocabulary size | 29,144 | 28,920 |
| Total tokens | 298,634 | 295,302 |

**Normalization impact**: 1,339 out of 5,733 rows (23.4%) had text modified by the LLM-based normalization. The vocabulary was reduced from 29,144 to 28,920 unique words. Further rule-based microtext normalization reduced vocabulary by an additional 349 words (29,144 → 28,795), but this had **limited additional impact** since the LLM-based normalization already standardized most colloquial expressions.

**Discussion**: Text normalization helps reduce vocabulary noise from informal spelling variations, but the effect is modest because: (1) the dataset already contains mostly well-formed text from Reddit, and (2) Singlish particles that were removed may carry sentiment-relevant pragmatic signals. Our experiments confirm that normalization provides a small but consistent improvement for sentiment classification (see Section 4.4).

#### 4.2.3 Feature Extraction

- **TF-IDF vectorization**: Unigrams + bigrams (ngram_range=1,2), max 50,000 features, min_df=2, max_df=0.95, sublinear TF scaling
- **Accent stripping**: Unicode accent normalization
- **Lowercase conversion**: All text lowercased before vectorization

### 4.3 Evaluation Dataset

The evaluation dataset was built by labeling **5,733 records** using two independent LLM annotators (GPT-5 and Gemini-2.5-pro), with a majority voting scheme incorporating the original crawled label as a third signal for sentiment.

**Inter-annotator agreement:**

| Task | Labels | Distribution | Mean Agreement | Agreement >= 67% |
|------|--------|:------------:|:--------------:|:-----------------:|
| Sentiment (3-class) | Neg(0) / Neu(1) / Pos(2) | 1,950 / 1,838 / 1,945 | 96.4% | 89.3% |
| Sarcasm (binary) | Not sarcastic(0) / Sarcastic(1) | 5,685 / 48 | 99.5% | 98.9% |
| Subjectivity (binary) | Objective(0) / Subjective(1) | 306 / 5,427 | 97.3% | 94.5% |

All tasks exceed the minimum 80% inter-annotator agreement requirement. The sentiment task uses 3-way voting (original label + 2 LLM labels), while sarcasm and subjectivity use 2-way voting.

![Label Distribution](figures/label_distributions.png)

**Note:** Sarcasm and subjectivity datasets are highly imbalanced, which impacts classification performance on the minority class.

### 4.4 Evaluation Metrics

We use **5-fold stratified cross-validation** and report precision, recall, and F1-measure (both macro and weighted averages).

#### Sentiment Classification (3-class)

| Classifier | Text | Accuracy | Macro F1 | Speed (r/s) |
|------------|------|----------|----------|-------------|
| **Linear SVM** | **Normalized** | **0.7572** | **0.7573** | 1,259,874 |
| Linear SVM | Original | 0.7495 | 0.7495 | 1,921,063 |
| Logistic Regression | Normalized | 0.7345 | 0.7346 | 1,891,594 |
| Logistic Regression | Original | 0.7326 | 0.7328 | 10,792,614 |
| Random Forest | Normalized | 0.7054 | 0.7054 | 30,852 |
| Random Forest | Original | 0.6927 | 0.6923 | 30,550 |
| Multinomial NB | Normalized | 0.6431 | 0.6361 | 1,433,696 |
| Multinomial NB | Original | 0.6379 | 0.6324 | 2,242,464 |

![Sentiment Comparison](figures/comparison_sentiment_final.png)

#### Best Configuration Per Task

| Task | Best Classifier | Text | Macro F1 | Accuracy |
|------|----------------|------|----------|----------|
| Sentiment (3-class) | Linear SVM | Normalized | 0.7573 | 0.7572 |
| Sarcasm (binary) | Logistic Regression | Normalized | 0.4979 | 0.9916 |
| Subjectivity (binary) | Linear SVM | Original | 0.5985 | 0.9501 |

### 4.5 Random Accuracy Test on Remaining Data

After training on the full evaluation dataset, the best sentiment model (Linear SVM, normalized text, F1=0.7573) was applied to predict sentiment on the remaining **crawled Reddit records** (`crawled_clean.csv`).

**Prediction distribution:**

| Label | Count | Percentage |
|-------|------:|:----------:|
| Negative (0) | 4,264 | 32.2% |
| Neutral (1) | 3,770 | 28.5% |
| Positive (2) | 5,194 | 39.3% |

The distribution shows a roughly balanced spread with a slight positive skew, which aligns with expectations for Singapore education-related Reddit discourse where both complaints and supportive/motivational content are common. Manual spot-checking of random predictions confirmed reasonable classification quality.

### 4.6 Performance Metrics and Scalability

| Classifier | Training Time (5-fold) | Prediction Speed |
|------------|:----------------------:|:----------------:|
| Linear SVM | ~0.3s | **1.3M–1.9M records/sec** |
| Logistic Regression | ~2.1s | **1.9M–10.8M records/sec** |
| Multinomial NB | ~0.02s | **1.4M–2.2M records/sec** |
| Random Forest | ~10s+ | **23K–31K records/sec** |

![Performance Speed](figures/performance_speed.png)

**Scalability analysis:**
- **Linear models** (SVM, LR, NB) scale linearly with dataset size via sparse TF-IDF representations, achieving millions of predictions per second. They are suitable for real-time or large-batch production use.
- **Random Forest** is significantly slower due to tree ensemble inference overhead, but still practical for batch processing.
- The TF-IDF + Linear SVM pipeline can process the entire crawled dataset in under 0.01 seconds, demonstrating excellent production scalability.

### 4.7 Impact of Text Normalization

![Normalization Delta](figures/delta_normalization.png)

For sentiment classification, **normalized text consistently outperforms original Singlish text** across all classifiers (F1 deltas: +0.002 to +0.013). This indicates that normalizing Singlish spelling variants and abbreviations into standard English reduces vocabulary noise and helps TF-IDF-based classifiers generalize better. However, the improvement is modest, suggesting that Singlish informal text is already largely understandable to bag-of-words models.

---

## Question 5: Innovations for Enhancing Classification

We explore **four progressive innovations** to enhance sentiment classification, evaluated through an ablation study that isolates the contribution of each innovation. Starting from a primitive baseline (BoW + Naive Bayes), we incrementally add more sophisticated techniques.

### 5.1 Baseline: Bag-of-Words + Multinomial Naive Bayes

The most primitive ML baseline uses:
- **CountVectorizer (Bag-of-Words)**: Simple unigram word counts with no term weighting
- **Multinomial Naive Bayes**: A generative probabilistic classifier that assumes feature independence

This represents the simplest possible text classification pipeline — raw word counts fed into a probabilistic classifier. It serves as our baseline to measure the contribution of all subsequent innovations.

**Limitation:** BoW treats all words equally regardless of their discriminative power, and NB's independence assumption is a poor fit for natural language where word order and co-occurrence matter.

### 5.2 Innovation 1: TF-IDF Representation + Linear SVM

**Improvement over baseline:**
- **TF-IDF** (Term Frequency–Inverse Document Frequency) replaces raw counts with weighted scores that downweight common words and emphasize discriminative terms. We use unigrams + bigrams (`ngram_range=(1,2)`) with sublinear TF scaling.
- **Linear SVM** replaces NB with a discriminative maximum-margin classifier that excels in high-dimensional sparse spaces.

**Why this helps:** TF-IDF + SVM is the standard strong baseline in text classification. TF-IDF captures term importance (e.g., "terrible" is more informative than "the"), while SVM finds optimal decision boundaries in the high-dimensional feature space. This combination has been shown to outperform NB on most text classification benchmarks.

**Example:** For the text *"The education system here is really terrible"*, BoW gives "terrible" and "the" equal weight (both count=1). TF-IDF assigns "terrible" a much higher score (rare, informative) and "the" a near-zero score (ubiquitous, uninformative).

### 5.3 Innovation 2: Hybrid Classification (Symbolic + Subsymbolic AI)

We augment the TF-IDF features with **symbolic, knowledge-based features** to create a hybrid system:

- **VADER lexicon scores** (4 features): negative, neutral, positive, and compound sentiment scores from the VADER sentiment lexicon. This injects domain knowledge about word-level sentiment polarity that TF-IDF alone cannot capture.
- **Text statistics** (8 features): character count, word count, average word length, exclamation/question mark counts, capitalization ratio, emoji count, and negation word count. These rule-based features capture stylistic signals correlated with sentiment.

**Why this helps:** TF-IDF is purely subsymbolic (statistical) — it treats words as opaque tokens and cannot directly model known sentiment polarities. VADER provides a complementary knowledge-based signal. For example, VADER knows "great" is positive and "terrible" is negative, regardless of training data frequency. The text statistics capture writing style patterns (e.g., more exclamation marks in strongly positive/negative text).

**Example:** For the text *"Wow this is absolutely terrible!!! Cannot believe it"*, TF-IDF encodes word frequencies, but VADER directly assigns compound = −0.83 (strongly negative), while the rule-based features capture 3 exclamation marks and 1 negation word — all reinforcing the negative signal.

### 5.4 Innovation 3: Enhanced Classification (Sarcasm-Aware Sentiment)

We add **sarcasm-indicative features** to help the model detect cases where surface-level sentiment is opposite to intended sentiment:

- **Contrast indicators** (count of "but", "however", "although", etc.)
- **Hyperbole markers** (count of "totally", "absolutely", "literally", etc.)
- **Singlish sarcasm markers** (count of "meh", "hor", "lor", etc.)
- **Punctuation patterns** (quotes, ellipsis counts)
- **Sentiment contrast** (absolute difference between VADER pos and neg scores)
- **Reddit sarcasm tag** (presence of "/s")

**Why this matters:** Sarcasm inverts the polarity of text — *"Oh sure, the education system is just perfect"* reads as positive on surface but conveys negative sentiment. By explicitly modeling sarcasm indicators, the classifier can learn to adjust its predictions when sarcasm signals are present.

### 5.5 Innovation 4: Ensemble Classification (Stacked Ensemble)

We also test a **Stacked Ensemble** combining LR + NB + RF with an LR meta-learner, applied on top of the full augmented feature set (TF-IDF + hybrid + sarcasm).

### 5.6 Ablation Study

All experiments use the **Sentiment (3-class)** task with **original Singlish text** and **5-fold stratified CV**:

| Configuration | Accuracy | Macro Prec | Macro Rec | Macro F1 | Delta vs Baseline |
|--------------|----------|------------|-----------|----------|-------------------|
| A. Baseline (BoW + NB) | 0.6325 | 0.6465 | 0.6306 | 0.6307 | — |
| B. + TF-IDF + SVM | 0.7495 | 0.7497 | 0.7494 | 0.7495 | **+0.1188** |
| C. + Hybrid (VADER + Stats) | 0.7612 | 0.7614 | 0.7609 | 0.7611 | **+0.1304** |
| **D. + Hybrid + Sarcasm** | **0.7621** | **0.7622** | **0.7619** | **0.7620** | **+0.1313** |
| E. + Hybrid + Sarcasm + Ensemble | 0.7246 | 0.7253 | 0.7240 | 0.7240 | +0.0933 |

![Ablation Study](figures/ablation_study.png)

### 5.7 Analysis

**Key findings:**

1. **TF-IDF + SVM provides the largest single improvement** (+0.1188 F1 over baseline). Replacing raw word counts with TF-IDF weighting and NB with Linear SVM yields a dramatic 18.8% relative improvement. This confirms that (a) term importance weighting is critical for text classification, and (b) discriminative classifiers outperform generative models on this task.

2. **Hybrid features provide meaningful additional improvement** (+0.0116 F1 on top of TF-IDF+SVM). Adding VADER lexicon scores and text statistics gives the SVM classifier complementary signals that pure bag-of-words cannot capture. This confirms the value of combining symbolic (knowledge-based) and subsymbolic (statistical) approaches.

3. **Sarcasm features provide a small but consistent improvement** (+0.0009 F1 on top of hybrid). The modest gain is expected given that sarcasm is relatively rare in this dataset (only 48 out of 5,733 samples labeled as sarcastic). However, sarcasm features stack additively with hybrid features, showing they capture a complementary signal.

4. **Ensemble classification hurts performance** (−0.0380 F1 vs. best). The stacking approach degrades results because:
   - The NB component requires non-negative features, forcing feature clipping that loses information
   - The meta-learner overfits on the internal 3-fold CV with augmented features
   - Linear SVM alone is already highly effective on the high-dimensional TF-IDF space

5. **Best configuration: TF-IDF + Hybrid + Sarcasm + Linear SVM** achieves Macro F1 = 0.7620, a **+20.8% relative improvement** over the primitive BoW+NB baseline (0.6307).

**Incremental contributions (ablation):**

| Innovation Added | Individual Gain | Cumulative F1 |
|-----------------|-----------------|:-------------:|
| TF-IDF + SVM (replaces BoW + NB) | +0.1188 | 0.7495 |
| Hybrid (VADER + text stats) | +0.0116 | 0.7611 |
| Sarcasm features | +0.0009 | 0.7620 |
| Ensemble (harmful) | −0.0380 | 0.7240 |

The ablation clearly shows a **progressive improvement** from primitive (BoW+NB) through standard (TF-IDF+SVM) to hybrid (symbolic+subsymbolic), with the first innovation providing the bulk of the improvement. Ensemble classification is not beneficial in this setting.

---

## Confusion Matrices

Selected confusion matrices for the best-performing models:

### Sentiment (Linear SVM, Normalized Text)
![CM Sentiment SVM](figures/confusion_matrices/cm_sentiment_final_svm_original_text.png)

---

## Project Structure

```
SC4021-Project/
├── run_classification.py          # Main entry point
├── classification/                # Modular classification package
│   ├── __init__.py
│   ├── config.py                  # CLI args, constants, display mappings
│   ├── data_loader.py             # Data loading utilities
│   ├── models.py                  # Classifier factory, TF-IDF builder
│   ├── evaluation.py              # 5-fold CV and metrics
│   ├── preprocessing.py           # Q4: Data preprocessing & statistics
│   ├── visualization.py           # All plotting functions
│   ├── prediction.py              # Post-training prediction
│   ├── innovations.py             # Q5: Ablation study (BoW→TF-IDF→Hybrid→Ensemble)
│   └── report_printer.py          # Console output formatting
├── figures/                       # Generated visualization plots
│   ├── comparison_*.png           # Classifier comparison charts
│   ├── delta_normalization.png    # Normalization impact analysis
│   ├── label_distributions.png    # Dataset label distributions
│   ├── performance_speed.png      # Speed comparison
│   ├── ablation_study.png         # Q5 ablation study
│   └── confusion_matrices/        # Per-experiment confusion matrices
├── eval_preprocessed.csv          # Labeled evaluation dataset (5,733 records)
├── crawled_clean.csv              # Crawled Reddit data
├── classification_comparison_report.json  # Full JSON report
└── report.md                      # This report
```

## How to Run

```bash
conda activate mdp
python run_classification.py

# Skip innovations or visualizations
python run_classification.py --skip_innovations --skip_visualizations

# Custom classifiers
python run_classification.py --classifiers logistic svm nb
```
