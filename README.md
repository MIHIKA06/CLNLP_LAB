# CLNLP_LAB

Lab work for **Computational Linguistics & Natural Language Processing (CLNLP)**: Mihika, SAP ID 500122055.

The repository has one Jupyter notebook per experiment (Experiment 9 has its own folder). The full write-up of all 10 experiments, with an abstract, a code description, the implemented code and the results and graphs for each, is in the lab report:

- [CLNLP_Lab_Report_Exp1-10_MIHIKA_500122055.pdf](CLNLP_Lab_Report_Exp1-10_MIHIKA_500122055.pdf) (editable version: [.docx](CLNLP_Lab_Report_Exp1-10_MIHIKA_500122055.docx))

## Experiments at a glance

| # | Title | Main topics | Notebook |
|---|---|---|---|
| 1 | Data Analysis and Visualization | pandas group-by and filtering; bar, pie, histogram and scatter plots | [Experiment_1.ipynb](Experiment_1.ipynb) |
| 2 | Basic Text Preprocessing | text cleaning; tokenization with regex, NLTK and spaCy; stop-word removal | [Experiment_2.ipynb](Experiment_2.ipynb) |
| 3 | Stemming, Lemmatization and Regex | Porter stemmer, WordNet lemmatizer, regex extraction of emails, URLs, phone numbers, hashtags and mentions | [Experiment_3.ipynb](Experiment_3.ipynb) |
| 4 | Term Frequency, NER and TF-IDF | term frequency with and without toolkits, spaCy NER, TF-IDF from scratch | [Experiment_4.ipynb](Experiment_4.ipynb) |
| 5 | Subword Tokenization, POS Tagging and Word2Vec | BPE, SentencePiece, POS tagging, CBOW, Skip-Gram, cosine similarity | [Experiment_5.ipynb](Experiment_5.ipynb) |
| 6 | GloVe Embeddings and N-gram Language Model | GloVe from scratch in PyTorch and pretrained, bigram next-word prediction | [Experiment_6.ipynb](Experiment_6.ipynb) |
| 7 | IMDB Sentiment Analysis: Model Comparison | Logistic Regression, LSTM, vanilla Transformer, BERT | [experiment_7.ipynb](experiment_7.ipynb) |
| 8 | Multilingual Abstractive Summarization | mT5 vs mBART on Hindi XL-Sum, ROUGE and BERTScore | [Experiment_8.ipynb](Experiment_8.ipynb) |
| 9 | Mini Project: Hinglish Cyberbullying Detection | 7-class classification, TF-IDF + NB/LR/SVM, fine-tuned MuRIL | [Experiment_9/](Experiment_9/README.md) |
| 10 | Project: English–Khasi Neural Machine Translation | Seq2Seq LSTM, Transformer, NLLB-200 + LoRA, BLEU/ChrF++/METEOR/TER | lab report only |

## Experiment details

### Experiment 1: Data Analysis and Visualization
An employee dataset of 100 records (department, salary, gender, age, experience) is analysed with pandas and visualised with matplotlib.
- **Tasks:** average salary and employee count per department (bar charts), gender split (pie chart), salary and age distributions (histograms), experience vs salary (scatter plot), top 10 earners, highest salary per department, employees above the average salary, and average experience per department.
- **Key methods:** `groupby().mean()/max()`, `value_counts()`, `nlargest()`, boolean filtering.
- **Result:** Operations has the highest average salary, HR has the most employees (21), and the workforce is 64% male and 36% female.

### Experiment 2: Basic Text Preprocessing
- **2.1 Cleaning:** lowercases the text, removes punctuation, digits and extra whitespace, and reports what was removed (32 uppercase letters, 6 punctuation types, 13 digits, 12 extra spaces).
- **2.2 Tokenization:** words and sentences are split three ways: with regex (`\w+`, `[.!?]+`), with NLTK (`word_tokenize`, `sent_tokenize`) and with spaCy (`en_core_web_sm`).
- **2.3 Stop-words:** NLTK's English stop-word list separates stop-words from content tokens.

### Experiment 3: Stemming, Lemmatization and Regex Information Extraction
- **3.1 Porter stemming:** rule-based suffix removal, which can produce non-words (studies → studi, computers → comput).
- **3.2 WordNet lemmatization:** dictionary-based base forms. With the correct part of speech it gives real words (ran → run, better → good, mice → mouse).
- **3.3 Regex:** patterns extract email addresses, URLs, mobile numbers, hashtags and @mentions from a paragraph.

### Experiment 4: Term Frequency, Named Entity Recognition and TF-IDF
- **4.1:** term frequency with NLTK and `Counter`, saved to CSV with the top 10 terms, plus named entities (ORG, GPE, LANGUAGE, …) found with spaCy.
- **4.2:** the same term frequency in plain Python, using a dictionary and manual CSV writing.
- **4.3:** TF-IDF from scratch on 3 documents: TF = count / length, DF, IDF = log10(N/DF), and TF-IDF = TF × IDF, with the top terms for each document.

### Experiment 5: Subword Tokenization, POS Tagging and Word2Vec
- **5.1 BPE:** the pretrained GPT-2 tokenizer (1,348 tokens) vs a BPE model trained from scratch with a 1,000-token vocabulary (1,111 tokens).
- **5.2 SentencePiece:** the pretrained T5 tokenizer (1,319 tokens) vs a SentencePiece model trained from scratch (1,198 tokens).
- **5.3 and 5.4 POS tagging:** spaCy and NLTK tags with descriptions, plus tag frequencies over the whole text.
- **5.5 to 5.7 Word2Vec:** CBOW and Skip-Gram models trained with gensim (50 dimensions, window 3) on a 10-sentence corpus. The cosine similarity of (machine, learning) is 0.483.

### Experiment 6: GloVe Word Embeddings and N-gram Language Model
- **GloVe from scratch (PyTorch):** builds a co-occurrence matrix (window 2, weighted by 1/distance), then optimises the weighted least-squares GloVe loss with Adagrad for 300 epochs.
- **Pretrained GloVe** (`glove-wiki-gigaword-50` via gensim): the words most similar to "king" are prince (0.82), queen (0.78) and emperor (0.77). The from-scratch model trained on 24 sentences gives poor neighbours, which shows that embeddings need lots of data.
- **Bigram language model:** P(w2 | w1) is estimated from counts and used to predict the next word (natural → language, machine → learning).

### Experiment 7: IMDB Sentiment Analysis: Model Comparison
Four classifiers are trained on a balanced subset of 2,000 training and 500 test reviews:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.838 | 0.744 | 0.788 |
| LSTM (from scratch) | 0.521 | 0.780 | 0.625 |
| Vanilla Transformer (from scratch) | 0.757 | 0.524 | 0.619 |
| **BERT (fine-tuned)** | 0.829 | 0.908 | **0.866** |

Pretrained BERT performs best. The from-scratch neural models need more data and epochs than this subset gives them.

### Experiment 8: Multilingual Abstractive Summarization (mT5 vs mBART)
Hindi news articles from XL-Sum (10 test articles) are summarised with beam search (4 beams, max length 84):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore (mBERT F1) |
|---|---|---|---|---|
| **mT5 (fine-tuned on XL-Sum)** | **0.313** | **0.137** | **0.262** | **0.761** |
| mBART-large-50 | 0.196 | 0.052 | 0.151 | 0.700 |

mT5 wins because it was fine-tuned for summarization, while mBART-large-50 is a general pretrained model. ROUGE uses a whitespace tokenizer because the default tokenizer drops Devanagari characters.

### Experiment 9: Mini Project: Cyberbullying Detection in Hinglish Social Media
**Team:** Mihika, Yash Parmar, Diptanshu Banerjee

Seven-class classification (not_cyberbullying, gender, religion, age, abusive, offensive, mockery) on the MC-Hinglish1.0 dataset (7,400 training and 1,000 test posts), using word and character TF-IDF features with Naive Bayes, Logistic Regression and Linear SVM, plus a fine-tuned MuRIL transformer. The **Linear SVM is the best model, with 75.2% accuracy and 0.736 macro F1**. MuRIL reached 67.1% accuracy. The full write-up, results and run instructions are in [Experiment_9/README.md](Experiment_9/README.md).

### Experiment 10: Project: English–Khasi Neural Machine Translation
Three systems for English→Khasi, a low-resource language pair from the WMT26 Low-Resource Indic NMT task: a Seq2Seq LSTM, a Transformer trained from scratch, and NLLB-200 fine-tuned with LoRA. The pipeline covers data collection, preprocessing, SentencePiece tokenization, training and evaluation.

| Model | BLEU | ChrF++ | METEOR | TER (lower is better) |
|---|---|---|---|---|
| Seq2Seq LSTM | **1.36** | **13.48** | **17.23** | 157.67 |
| Transformer (scratch) | 0.04 | 1.26 | 3.39 | 181.17 |
| NLLB-200 + LoRA | 0.11 | 5.09 | 2.65 | **101.89** |

All systems score low. The from-scratch models collapse into repeated tokens, and NLLB-LoRA mostly copies the English input, which shows that the training data is too small for this language pair. The code for this project is not in this repository; its write-up, code and graphs are in the lab report.

## Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab or Google Colab (a GPU is recommended for Experiments 7, 8 and 9 (MuRIL))

```bash
pip install pandas numpy matplotlib seaborn nltk spacy scikit-learn gensim
pip install torch transformers datasets tokenizers sentencepiece evaluate rouge-score bert-score
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Experiments 1, 2, 4 and 5 read their input files from a mounted Google Drive folder (`/content/drive/MyDrive/CNNLP_LAB/...`) when run on Colab. Update the file path in the notebook if you run it elsewhere. Experiments 7 and 8 download their datasets from Hugging Face automatically, and Experiment 9 includes its data in `Experiment_9/data/`.

## How to Run

1. Clone the repository and `cd` into it.
2. Install the prerequisites above.
3. Start Jupyter with `jupyter notebook` (or open a notebook in Google Colab) and open the experiment you want.
4. Run the cells from top to bottom. Later cells depend on variables defined earlier, such as `df` and `text`.
