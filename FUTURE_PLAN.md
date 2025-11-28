# Future Plan

This document outlines the current status of the project (Phase 1) and proposes future enhancements for the next phase (Phase 2).

## Phase 1: Completed

The following features have been successfully implemented and documented:

*   **Core Chatbot Logic**:
    *   Natural Language Processing (NLP) pipeline using NLTK (tokenization, stemming).
    *   Bag-of-words representation for text input.
    *   Deep Neural Network (DNN) classification model using TFLearn/TensorFlow.
*   **Web Server**:
    *   Flask application serving a web interface.
    *   API endpoints for integrating the chatbot into other applications (`/response`).
*   **Data Management**:
    *   JSON-based training data storage (`data_to_train.json`).
    *   Pickle serialization for processed training data optimization.
*   **Documentation**:
    *   Comprehensive docstrings for all source code.
    *   Detailed `README.md` with setup and usage instructions.

## Phase 2: Future Enhancements

The following features are proposed for the next development phase:

### 1. Improved NLP & Model Architecture
*   **Lemmatization**: Switch from Stemming (Lancaster) to Lemmatization (WordNet) for better linguistic accuracy.
*   **Word Embeddings**: Implement Word2Vec, GloVe, or BERT embeddings instead of a simple bag-of-words model to capture semantic meaning.
*   **Context Awareness**: Add state management to handle multi-turn conversations and context retention (e.g., remembering user's name or previous topic).
*   **Entity Extraction**: Implement Named Entity Recognition (NER) to extract specific information (dates, locations, names) from user input.

### 2. Backend & Infrastructure
*   **Database Integration**: Move `data_to_train.json` to a database (SQL or NoSQL) to allow dynamic updates without restarting the server.
*   **Model Versioning**: Implement a system to manage multiple versions of the trained model.
*   **Dockerization**: Containerize the application using Docker for consistent deployment across environments.
*   **Asynchronous Processing**: Use Celery or similar tools if processing becomes heavy, ensuring the web server remains responsive.

### 3. Frontend Improvements
*   **Modern UI Framework**: Rewrite the frontend using React, Vue, or Angular for a more dynamic and responsive user experience.
*   **Real-time Communication**: Switch from HTTP polling/reloading to WebSockets (using Flask-SocketIO) for real-time chat updates.
*   **Chat History Persistence**: Store chat logs in a database or local storage so history isn't lost on refresh.

### 4. Code Quality & Testing
*   **Unit Tests**: Add a test suite (using `pytest` or `unittest`) to verify intent classification and API responses.
*   **CI/CD Pipeline**: Set up a Continuous Integration/Continuous Deployment pipeline (e.g., GitHub Actions) to automate testing and deployment.
*   **Type Hinting**: Add Python type hints to improve code readability and allow for static type checking.
