# AI ChatBot

A Python-based AI Chatbot that uses Deep Learning (TensorFlow/TFLearn) and Natural Language Processing (NLTK) to understand and respond to user queries. It provides both a command-line interface and a Flask web interface.

## Description

This project implements a retrieval-based chatbot. It works by:
1.  **Preprocessing**: Tokenizing and stemming user input.
2.  **Classification**: Using a Deep Neural Network (DNN) to classify the intent of the input based on trained patterns.
3.  **Response**: Selecting a random response associated with the identified intent from a predefined dataset (`data_to_train.json`).

The application can be run as a web server, providing a simple chat interface and API endpoints.

## Project Structure

-   `server.py`: The main entry point. Handles data loading, model training/prediction, and the Flask server.
-   `data_to_train.json`: A JSON file containing the training data (intents, patterns, and responses).
-   `templates/index.html`: The HTML template for the web interface.
-   `static/styles.css`: CSS styles for the web interface.
-   `data.pickle`: Serialized preprocessed training data (generated automatically).
-   `model.tflearn.*`: Saved model files (generated automatically after training).

## Getting Started

### Prerequisites

-   Python 3.6+
-   pip (Python package manager)

### Installation

1.  **Clone the repository** (if applicable) or download the source code.

2.  **Create a Virtual Environment** (Recommended)
    It is best practice to run this project in a virtual environment to avoid conflicting dependencies.

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install the required Python packages using `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you encounter issues installing TensorFlow or TFLearn, ensure your Python version is compatible with the versions specified or the latest available versions.*

4.  **Download NLTK Data**
    The project uses the 'punkt' tokenizer. You might need to download it if not already present.

    ```python
    import nltk
    nltk.download('punkt')
    ```
    (This is often handled automatically or can be run once in a Python shell).

## Usage

### Running the Application

To start the Flask server and run the chatbot:

```bash
python server.py
```

This will:
1.  Load the training data.
2.  Train the model (if `model.tflearn` does not exist) or load the existing model.
3.  Start the web server on `http://localhost:5000`.

### Web Interface

Open your web browser and navigate to `http://localhost:5000`. You will see a chat interface where you can type messages and receive responses from the bot.

### API Endpoints

The server exposes the following endpoints:

-   **`GET /`**: Renders the chat interface.
-   **`POST /`**: Accepts a message form submission and updates the chat.
-   **`POST /response`**:
    -   **Body**: Form data with key `msg`.
    -   **Return**: Plain text response.
-   **`GET /response/<msg>`**:
    -   **Parameter**: `msg` (the message string).
    -   **Return**: Plain text response.

    *Example:* `http://localhost:5000/response/hello`

## Training the Model

The model is trained based on the contents of `data_to_train.json`.

**To retrain the model:**
1.  Modify `data_to_train.json` with new intents, patterns, or responses.
2.  Delete the existing `model.tflearn` files (index, meta, data) and `data.pickle`.
3.  Run `python server.py`. The script will detect the missing model/data and retrain using the updated JSON file.

### data_to_train.json Format

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day"],
      "responses": ["Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?"]
    },
    ...
  ]
}
```
