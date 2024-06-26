# NLP Chatbot Project

Welcome to the NLP Chatbot Project! This project is a demonstration of Natural Language Processing (NLP) techniques, featuring a small chatbot with a Gradio UI interface. The chatbot can respond to questions based on a given context and previously learned data. It includes a small GPT-like model trained with a dictionary and several hundred rows of data.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

These instructions will help you set up and run the project on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/nlp-chatbot-project.git
   cd nlp-chatbot-project
   ```
2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
## Usage

To run the chatbot, execute the following command:

```bash
python ui.py
```

This will launch the Gradio UI interface where you can input a specific 'context' and a 'question'. The chatbot will then provide an answer based on the context and the previously learned data.

### Model
The project includes a small GPT-like model that has been trained with a dictionary and several hundred rows of data. This model is used to generate responses based on the input context and question.
