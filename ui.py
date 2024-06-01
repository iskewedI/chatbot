from keras.saving import load_model
import gradio as gr
import json

from utils import create_vectorization_from_input, parse_result_from_model

# Load and instantiate keras model
model = load_model("saved_model/chatbot_200_epochs.keras")

# Load tokenizer data
with open("saved_model/tokenizer.json") as f:
    tokenizer_data = json.load(f)

def predict(story, question):
    # We send "yes" by default as expected_answer but it doesn't make any difference
    story_vec, question_vec, _ = create_vectorization_from_input(story, question, "yes", tokenizer_data)

    pred_results = model.predict([story_vec, question_vec])

    result, certainty = parse_result_from_model(pred_results, tokenizer_data["word_index"])

    print(f"Predicted with {certainty} certainty")

    return result

with gr.Blocks() as block:
    gr.Markdown("Write the story and the question and receive an answer.")

    with gr.Row():
        story_in = gr.Textbox(label="Story", placeholder="Story context")
        question_in = gr.Textbox(label="Question", placeholder="What's the question?")

    predict_btn = gr.Button("Send message")

    output = gr.Textbox()

    predict_btn.click(fn=predict, inputs=[story_in, question_in], outputs=output)

block.launch()