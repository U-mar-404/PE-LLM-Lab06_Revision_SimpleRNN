# Lab 06 — Text Classification with SimpleRNN

IMDB movie review sentiment classification using a `SimpleRNN` in Keras/TensorFlow, built step by step as a revision lab before moving to LSTM.

**Course:** Prompt Engineering and LLM Applications | BSAI 7th Semester
**Instructor:** Iqra Riaz

## Pipeline

```
Text → IDs → Padding → Embedding → SimpleRNN (h1…hT) → Dense + Sigmoid → Probability → Class
```

## Model

```
Embedding(10000 → 128) → SimpleRNN(128) → Dropout(0.5) → Dense(1, sigmoid)
```

| Setting | Value |
|---|---|
| Dataset | IMDB (25,000 train / 25,000 test, balanced) |
| Vocabulary size | 10,000 |
| Max sequence length | 200 (pre-padding) |
| Optimizer / Loss | Adam / Binary cross-entropy |
| Epochs / Batch size | 5 / 64 |

## Steps Covered

1. Text → numbers (integer IDs)
2. Padding and truncation
3. Embedding layer
4. RNN equation: `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`
5. From final hidden state to decision (Dense + Sigmoid)
6. Building the model and counting parameters
7. Training (forward pass, loss, BPTT, Adam)
8. Reading results (curves, classification report, confusion matrix)
9. Predicting custom sentences
10. Verifying Keras `SimpleRNN` against a manual NumPy implementation
11. Hyperparameter experiments

## Results

**Test accuracy: 79.12%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.83 | 0.73 | 0.78 |
| Positive | 0.76 | 0.86 | 0.80 |

Confusion matrix: TP = 10702, TN = 9079, FP = 3421, FN = 1798

**Manual vs Keras check:** max |difference| in `h_T` ≈ 2.3e-07, confirming `SimpleRNN` matches the equation exactly.

## Experiments (3 epochs each)

| Experiment | Setting | Test Acc | Time (s) |
|---|---|---|---|
| Baseline | len=200, emb=128, units=128 | 0.6004 | 95 |
| Length 50 | len=50, emb=128, units=128 | 0.7685 | 31 |
| Length 400 | len=400, emb=128, units=128 | 0.6708 | 187 |
| Embedding 32 | len=200, emb=32, units=128 | 0.5154 | 60 |
| Embedding 64 | len=200, emb=64, units=128 | 0.5139 | 61 |
| Units 32 | len=200, emb=128, units=32 | 0.8320 | 63 |
| Units 64 | len=200, emb=128, units=64 | 0.6859 | 59 |

Results are noisy: SimpleRNN training is unstable on long sequences (vanishing gradients), so short sequences and fewer units tended to do better. This is the motivation for LSTM in the next lab.

## Lab Questions and Answers

**1. Why can an RNN not process a raw text string?**
An RNN understands numbers, not words directly. So we convert words into numbers before giving them to the RNN.

**2. Why do we use an embedding layer?**
An embedding layer converts words into number vectors. It helps the model understand the meaning of words.

**3. What is the role of h<sub>t-1</sub>?**
h<sub>t-1</sub> is the hidden state from the previous time step. It carries information from earlier words to the current word.

**4. Why are the same recurrent weights reused at every time step?**
The same weights are used to process every word in a sentence. This reduces the number of parameters and keeps learning consistent.

**5. What is h<sub>0</sub> and why?**
h<sub>0</sub> is the initial hidden state, usually set to zeros. It gives the RNN a starting point before reading any words.

**6. Why is the final hidden state used for classification?**
The final hidden state contains information learned from the sequence. We use it to predict the sentence's class, such as positive or negative.

**7. Why sigmoid in the last layer?**
Sigmoid converts the output into a value between 0 and 1. This value can represent the probability of a positive or negative class.

## Setup

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
jupyter notebook Lab06_Revision_SimpleRNN.ipynb
```

Or open the notebook directly in Google Colab.

## Files

- `Lab06_Revision_SimpleRNN.ipynb` — full notebook with code, outputs, and lab questions
