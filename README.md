# Music Artist Classification & Noise Robustness Analysis

An experimental Music Information Retrieval (MIR) and audio classification study evaluating how spectral and harmonic features perform in recognizing different music artists/genres, and how background noise and audio denoising affect classification accuracy.

## Repository Structure

* `main_pipeline.m`: High-level MATLAB orchestration script managing dataset partitioning, feature extraction calls, kNN hyperparameter selection, noise injection (`SNR = 5 dB`), and post-denoising evaluation.

> **Note on Reproducibility:** Low-level DSP helper functions and raw audio tracks (`.mp3`/`.wav`) are excluded from this public repository due to audio copyright restrictions and academic integrity policies.

---

## Tech Stack & Methodology

* **Environment:** MATLAB (Signal Processing & Audio Toolbox)
* **Classifier:** k-Nearest Neighbors (kNN) evaluated across $k \in \{1, 5, 10, 20\}$
* **Windowing:** Fixed frame segmentation (`windowLength = 30 ms`, `stepLength = 10 ms`)

### Dataset
A balanced dataset of **18 audio tracks** (6 per artist, split 50/50 into training and test sets) representing three contrasting musical styles:
* **Laufey** – Modern Jazz / Vocal
* **Simple Plan** – Pop Punk / Rock
* **Stray Kids (SKZ)** – K-Pop (multi-genre blend of hip-hop, pop, and rock)

### Experimental Pipeline
1. **Feature Extraction:** Extracted **Chromagrams** (12 pitch classes capturing harmonic/melodic content), **MFCCs** (Mel-Frequency Cepstral Coefficients capturing short-term spectral envelope and timbre), and a **Combined (General)** feature vector concatenating both.
2. **Baseline Evaluation:** Trained and tested kNN classifiers on clean audio across each feature set to identify the optimal $k$ and feature configuration (`bestKNN`).
3. **Noise Injection & Denoising:**
   * Added `babble.wav` background noise at **SNR = 5 dB** to evaluate model robustness under degraded acoustic conditions.
   * Applied audio enhancement (denoising) to the corrupted signals and re-evaluated classification performance.

---

## Experimental Results

| Condition / Feature Set | Optimal $k$ | Laufey Recall | SKZ Recall | Simple Plan Recall | Key Findings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Chromagrams Only** | $10$ | `41.3%` | `40.6%` | `49.9%` | High confusion across all classes; SKZ was frequently misclassified as Simple Plan (`59.4%` error rate). |
| **2. MFCCs Only** | $10$ | `53.9%` | `30.0%` | `61.4%` | Timbral features boosted Simple Plan and Laufey accuracy, while 2,013 SKZ frames overlapped with Simple Plan. |
| **3. Combined (General)** | $20$ | `51.0%` | `40.3%` | `63.1%` | Best overall baseline accuracy (~51.4% global); Simple Plan reached its highest recognition rate. |
| **4. Noisy Audio (`SNR = 5 dB`)** | $20$ | `64.8%` | `42.1%` | `48.6%` | Additive babble noise boosted Laufey's recognition rate while degrading Simple Plan's accuracy. |
| **5. Enhanced (Denoised)** | $10$ | `41.3%` | `40.6%` | `49.9%` | Denoising restored Simple Plan as the top-predicted class, stabilizing overall accuracy around ~44–50%. |

---

## Key Takeaways

* **Genre Hybridity Impact:** K-Pop tracks (Stray Kids) blend hip-hop, rock, and electronic elements, resulting in high timbral and harmonic variance that frequently overlaps with Pop Punk (Simple Plan) in small-sample kNN spaces.
* **Timbre vs. Harmony:** MFCCs outperformed Chromagrams in distinguishing artists, confirming that spectral envelope (timbre) carries stronger discriminative weight than pitch-class distribution alone for these genres.
* **Non-Linear Noise Effects:** Adding mid-frequency babble noise (`5 dB` SNR) interfered heavily with dense, distorted instrumentation (Pop Punk) while inadvertently highlighting acoustic/vocal-dominant profiles (Jazz).
