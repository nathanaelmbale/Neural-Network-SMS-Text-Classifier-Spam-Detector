
# SMS Spam Detection Using Neural Networks

## Quick Overview 

**Problem:** Build a binary text classifier to distinguish between legitimate messages ("ham") and spam SMS using natural language processing and deep learning.

**Solution:** Implemented a neural network with text vectorization and word embeddings to classify SMS messages, achieving robust spam detection through sequential text processing.

**Impact:** Created a production-ready spam filter with high accuracy on unseen messages, demonstrating proficiency in NLP, text preprocessing, and TensorFlow text classification pipelines.

## Collabs Notebook
```
https://colab.research.google.com/drive/1pi-DqFmtXV82ZALjy22OBMJoUKM0yCQj?usp=drive_link
```


## Step-by-Step Implementation

### Step 1: Environment Setup & Library Installation

**What:** Installed a stable TensorFlow version and imported required libraries.

**Key Libraries:**
- TensorFlow / Keras for neural networks
- Pandas for data handling
- TensorFlow Datasets for utilities
- NumPy for numerical operations
- Matplotlib for visualization

**Version Control:** Ensured a stable TensorFlow installation by removing `tf-nightly` if present.

---

### Step 2: Data Acquisition

**What:** Downloaded the SMS Spam Collection dataset using `wget`.

**Dataset Files:**
- `train-data.tsv`: Training messages
- `valid-data.tsv`: Testing and validation messages

**Format:** Tab-separated values (TSV) with label and message columns.

**Source:** FreeCodeCamp project data repository.

---

### Step 3: Data Loading & Exploration

**What:** Loaded TSV files into Pandas DataFrames.

**Structure:**
- Column 1: Label (`ham` or `spam`)
- Column 2: Message text content

**Initial Analysis:** Examined the first few rows to understand data format and distribution.

**Key Finding:** Dataset is imbalanced, with significantly more ham than spam messages.

---

### Step 4: Label Encoding

**Problem:** The model requires numerical labels, but the dataset contains text labels.

**Solution:** Binary encoding of labels.

- `ham → 0` (legitimate message)
- `spam → 1` (spam message)

**Implementation:** Converted labels into NumPy arrays.

**Result:** Training and validation labels represented in binary format (0/1).

---

### Step 5: Text Preprocessing Pipeline

**Custom Standardization Function:** Implemented a text cleaning pipeline.

- Converted text to lowercase
- Removed HTML tags such as `<br />` using regular expressions
- Removed punctuation and special characters

**Purpose:** Normalize text variations (for example, "FREE" versus "free") for consistent processing.

**Impact:** Reduced vocabulary size and improved model generalization.

---

### Step 6: Text Vectorization Configuration

**Layer:** `TextVectorization` from Keras.

**Parameters:**
- `max_tokens = 10,000`: Vocabulary size limit
- `sequence_length = 120`: Fixed output length with padding and truncation
- `output_mode = 'int'`: Convert words to integer indices
- `standardize = custom_standardization`: Apply text cleaning function

**Vocabulary Building:** Adapted the vectorization layer on training messages only.

**Result:** Learned a word-to-index mapping based on the training corpus.

---

### Step 7: Message Vectorization

**Process:** Converted text messages into integer sequences.

- Training data: `train_messages → train_sequences`
- Test data: `test_messages → test_sequences`

**Example Output:**
- Original: `"Free money now!"`
- Vectorized: `[42, 187, 933, 0, 0, ...]` (padded to length 120)

**Benefit:** Enabled neural network processing of numerical sequences.

---

### Step 8: Neural Network Architecture Design

**Model Type:** Sequential neural network for binary text classification.

**Architecture:**
- Embedding layer: Converts word indices into 16-dimensional dense vectors
- Dropout layer (20%) for regularization
- GlobalAveragePooling1D layer to aggregate sequence information
- Dropout layer (20%) before final output
- Dense output layer with 1 neuron and sigmoid activation

**Why This Architecture:**
- Embeddings learn semantic word representations
- Global average pooling handles variable-length sequences
- Dropout reduces overfitting
- Sigmoid outputs probabilities between 0 (ham) and 1 (spam)

---

### Step 9: Model Compilation

**Optimizer:** Adam, suitable for NLP tasks due to adaptive learning rates.

**Loss Function:** Binary crossentropy, standard for binary classification.

**Metric:** Accuracy as the primary evaluation metric.

**Configuration:** Optimized for binary text classification.

---

### Step 10: Dataset Analysis

**Class Distribution Check:**
- Counted ham versus spam messages in the training set
- Calculated spam ratio percentage

**Purpose:** Identify class imbalance, which affects evaluation and interpretation.

**Finding:** Dataset contains more ham messages than spam.

**Implication:** Accuracy must be interpreted carefully due to imbalance.

---

### Step 11: Model Training

**Training Configuration:**
- Epochs: 60
- Validation data: Vectorized test sequences
- Verbose: 1 (progress bars enabled)

**Data Used:** Vectorized message sequences and binary labels.

**Monitoring:** Tracked accuracy and loss on both training and validation sets.

**Result:** Model learned patterns distinguishing spam from legitimate messages.

---

### Step 12: Post-Training Verification

**Immediate Testing:** Generated predictions for a sample spam message.

**Example:**
- "you have won £1000 cash! call to claim your prize."

**Vocabulary Check:** Verified that 10,000 words were learned.

**Sanity Check:** Confirmed output probabilities aligned with expectations.

**Purpose:** Ensure the model learned meaningful spam indicators such as "won," "prize," and "cash."

---

### Step 13: Model Evaluation

**Test Set Performance:** Evaluated on unseen validation data.

**Metrics Reported:**
- Test accuracy
- Test loss (binary crossentropy)

**Purpose:** Measure generalization to new messages.

**Result:** High accuracy indicating effective spam detection.

---

### Step 14: Model Persistence

**What:** Saved the trained model to disk.

**Format:** Keras native format (`.keras`).

**Filename:** `spam_model.keras`

**Benefit:** Enables model reuse without retraining.

**Use Case:** Deployment or continued development.

---

### Step 15: Prediction Function Implementation

**Function:** `predict_message(pred_text)`

**Input:** Single message string.

**Process:**
1. Vectorize input text using the trained `vectorize_layer`
2. Generate a probability prediction between 0 and 1
3. Apply threshold: ≥ 0.5 → spam, < 0.5 → ham

**Output Format:** `[probability_float, label_string]`

**Example:** `[0.9234, 'spam']` or `[0.1567, 'ham']`

**Key Design:** Returns both probability and label for interpretability.

---

### Step 16: Comprehensive Testing

**Test Suite:** Seven diverse messages covering edge cases.

**Examples Include:**
- Normal conversation
- Obvious spam
- Scheduling messages
- Service notifications
- Prize scams
- Casual reminders
- Personal stories

**Validation:** Compared predictions against expected labels.

**Output:** Detailed report with probability, prediction, and correctness.

**Result:**

✅ Successfully classified all test cases.

---

### Step 17: Final Validation

**Automated Test:** Built-in `test_predictions()` function.

**Pass Criteria:** All seven test messages correctly classified.

**Result:**

✅ *"You passed the challenge. Great job!"*

**Verification:** Model generalizes effectively across varied message types.

