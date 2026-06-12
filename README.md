# LSTM Next Word Predictor

This repository contains a deep learning-based next-word prediction web application. The application utilizes a Long Short-Term Memory (LSTM) recurrent neural network architecture to analyze text input and predict the most probable subsequent words. The user interface is built using Streamlit and features a dark theme, real-time predictions, interactive suggestion buttons, and probability analytics.

## Features

- **Real-Time Predictions**: The application runs model inference on user-typed text.
- **Interactive Suggestions**: Dynamic suggestion chips allow users to click recommended words to automatically append them to the text input box.
- **Probability Analytics**: Displays a confidence breakdown for the top predicted words using graphical progress bars.
- **Model Architecture Insights**: Includes a sidebar detailing the model metrics, parameters, and a layer-by-layer diagnostic summary.

## Project Structure

- `app.py`: Streamlit application script containing the user interface layout, styling, and prediction logic.
- `lstm_model.h5`: Trained Keras LSTM neural network model.
- `tokinizer.pkl`: Pickled Keras Tokenizer instance used for mapping words to sequence tokens.
- `max_len.pkl`: Pickled integer representing the maximum sequence length used during training and padding.

## Technical Details

- **Model Input Shape**: The embedding layer expects a sequence input of length 745.
- **Padding Configuration**: Input word token sequences are pre-padded to match the 745 context length before being passed to the network.
- **Softmax Probability**: The model outputs a probability distribution over the top vocabulary words (vocabulary size is determined by the tokenizer and dense output dimension of 10,000).

## Setup and Installation

### Prerequisites

Ensure you have Python 3.10 or higher installed. You will also need `conda` or a standard python environment with package management capabilities.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AmanSingh-404/LSTM-Next-Word-Predictor.git
   cd Next_Word_PREDICTION
   ```

2. **Set Up Python Environment**:
   If using Conda, you can create and activate a new environment:
   ```bash
   conda create -n next-word-env python=3.10
   conda activate next-word-env
   ```

3. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install tensorflow numpy streamlit
   ```

### Running the Application

Start the Streamlit application using the following command:
```bash
streamlit run app.py
```

The application will launch and output the access URLs:
- Local URL: `http://localhost:8501`
- Network URL: `http://<your-ip-address>:8501`

Open `http://localhost:8501` in your web browser to interact with the application.
