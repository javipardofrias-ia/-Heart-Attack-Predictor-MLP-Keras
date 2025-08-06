
# Heart-Attack-Predictor-MLP-Keras — MLP for Heart Attack Prediction

## 📘 Description

This project implements a Multilayer Perceptron (MLP) using the `Keras` library (part of `TensorFlow`) to predict the likelihood of a person having a heart attack based on various health-related features. The notebook serves as a practical example of a machine learning pipeline for a binary classification problem, demonstrating how to use neural networks on structured, real-world data to support medical diagnosis.

The project focuses on key steps such as data preprocessing, building a deep neural network architecture, and evaluating model performance using a confusion matrix.

## 🛠️ Requirements

Ensure you have Python and the following libraries installed:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

Libraries used:

  * `pandas` – For data loading and manipulation.
  * `numpy` – For numerical operations.
  * `keras` & `tensorflow` – The primary libraries for building and training the MLP model.
  * `scikit-learn` – For splitting the dataset (`train_test_split`).
  * `matplotlib.pyplot` & `seaborn` – For data visualization, particularly for plotting the confusion matrix.

## 📊 Data Preprocessing

Data preprocessing is a crucial step to prepare the `infarto.csv` dataset for the MLP. This project follows these steps:

1.  **Handling Missing Values**: The notebook specifically targets the `bmi` column, where rows with missing (`NaN`) values are removed. Any other `NaN` values in the dataset are filled with `0`.
2.  **Categorical Encoding**: Several columns representing categorical features (e.g., `edad`, `genero`, `casado`, `trabajo`, `residencia`, `uso_tabaco`, `hipertensión`, `enfermedad_coronaria`) are encoded into numerical float values.
3.  **Feature Selection**: The notebook mentions using a correlation matrix to identify and remove unnecessary columns for prediction, though the specific columns removed are not shown in the provided snippet.

## 🧠 AI Model Structure

The neural network is an MLP implemented as a `Sequential` model in Keras. Its architecture is as follows:

  * **Input Layer**: The model is defined to accept the preprocessed data, with the input shape matching the number of features in the dataset.
  * **Hidden Layers**: The model uses multiple `Dense` hidden layers. The architecture includes `Dropout` layers between the dense layers, which is a regularization technique to prevent overfitting by randomly dropping a fraction of neurons during training.
  * **Output Layer**: The final layer is a `Dense` layer with a single neuron, using a sigmoid activation function for binary classification, which outputs a probability between 0 and 1.
  * **Compilation**: The model is compiled with the `Adam` optimizer, which is an efficient optimization algorithm.

## 🧬 Training and Results

The training process focuses on optimizing the model's weights to minimize the loss and maximize accuracy.

  * **Training Loop**: The dataset is split into training and testing sets using `train_test_split`. The model is then trained on the training data.
  * **Performance Metrics**: After training, the notebook calculates and visualizes a confusion matrix to evaluate the model's performance on the test set. This metric helps to understand the number of true positives, true negatives, false positives, and false negatives, which is particularly important in a medical context.

## 🚀 How to Run

1.  Ensure you have the `infarto.csv` file in the same directory as the notebook.
2.  Open the notebook:
    ```bash
    jupyter notebook Practica3_Infarto (1).ipynb
    ```
3.  Run the cells sequentially to:
      * Import the necessary libraries.
      * Load and preprocess the dataset.
      * Define, compile, and train the MLP model.
      * Evaluate the model's performance and visualize the results.
