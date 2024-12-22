### README: Implement a Neural Network for Sentiment Analysis

---

#### **Project Description**
This project implements a neural network for sentiment analysis of book reviews from a dataset of Amazon reviews. The task is a binary classification problem to predict whether a review is positive or negative. The implementation uses Python, Keras, and Scikit-learn libraries for data preprocessing, model building, training, and evaluation.

---

#### **Key Steps**

1. **Load the Dataset**:
   - File: `data/bookReviews.csv`
   - The dataset contains two columns: `Review` (text) and `Positive_Review` (binary labels).

2. **Preprocess the Data**:
   - Extract `Review` as features (X) and `Positive_Review` as labels (y).
   - Split the dataset into training and testing sets with an 80/20 ratio using a random seed of 1234.

3. **Transform Text with TF-IDF**:
   - Use `TfidfVectorizer` to convert text data into numerical vectors.
   - Fit the vectorizer on training data and transform both training and test sets.

4. **Build the Neural Network**:
   - Architecture:
     - Input Layer: Matches the vocabulary size of the TF-IDF vectorizer.
     - Hidden Layers: Three layers with nodes (64, 32, 16) using ReLU activation.
     - Output Layer: Single node with sigmoid activation for binary classification.
   - Regularization:
     - Use dropout layers (e.g., `Dropout(0.25)`) to reduce overfitting.

5. **Compile and Train the Model**:
   - Loss Function: Binary cross-entropy.
   - Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.1.
   - Training:
     - Train for 50 epochs (or optimized number of epochs).
     - Use a validation split of 20% for monitoring overfitting.

6. **Evaluate Model Performance**:
   - Evaluate on the test set for accuracy and loss.
   - Use plots to visualize training/validation loss and accuracy over epochs.
   - Experiment with hyperparameters (e.g., epochs, dropout) to improve generalization.

7. **Make Predictions**:
   - Use the trained model to predict sentiment for the test set.
   - Apply a threshold of 0.5 to classify reviews as positive or negative.

---

#### **How to Run the Project**

1. Clone the repository and navigate to the project directory.
2. Ensure the required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```
3. Place the `bookReviews.csv` dataset inside a `data` folder in the project directory.
4. Execute the Python script or Jupyter notebook to process the dataset, train the model, and evaluate its performance.

---

#### **Expected Outputs**

1. Training and Validation Metrics:
   - Loss and Accuracy plots for training and validation data.
   - Model performance on the test set (loss and accuracy).

2. Example Predictions:
   - Probabilities and classifications for test reviews.
   - Comparison of predictions against actual labels for selected reviews.

---

#### **Key Findings**

- **Overfitting Reduction**:
  - Adding dropout layers reduced the gap between training and validation accuracy, improving generalization.
  - Increasing epochs beyond 50 led to overfitting; reducing to 10 epochs resulted in underfitting.
  - An optimal balance was achieved with ~50 epochs and dropout layers.

- **Accuracy**:
  - Training Accuracy: ~100%
  - Test Accuracy: ~82%

---

#### **Future Improvements**

- Use additional regularization techniques such as weight decay.
- Experiment with advanced architectures (e.g., LSTMs or Transformers) for better text representation.
- Increase dataset size for better generalization.

---

#### **Dependencies**

- Python 3.7+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow.keras

---

#### **License**

This project is open-source and free to use under the MIT License.


