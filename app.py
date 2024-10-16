import gradio as gr
from chatbot import Chatbot

# Initialize chatbot
chatbot = Chatbot()


# Define the Gradio interface
def chat_response(model_name, question, prompt):
    chatbot.set_model(model_name)
    return chatbot.generate_response(question, prompt)


# Default prompt examples
prompt_examples = [
    "Explain this in simple terms",
    "Provide more details",
    "Summarize the key points",
    "Explain it as if I'm a beginner"
]

model_options = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

with gr.Blocks() as demo:
    model_selector = gr.Dropdown(choices=model_options, label="Select Model", value=model_options[0])
    question = gr.Textbox(label="Question", placeholder="Type your question here...")
    prompt = gr.Textbox(label="Prompt", placeholder="Type your prompt here...")
    examples = gr.Examples(prompt_examples, inputs=prompt)
    output = gr.Textbox(label="Output")
    submit = gr.Button("Submit")

    submit.click(fn=chat_response, inputs=[model_selector, question, prompt], outputs=output)

if __name__ == "__main__":
    demo.launch()
