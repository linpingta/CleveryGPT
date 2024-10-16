from transformers import AutoModelForCausalLM, AutoTokenizer


class Chatbot(object):
    def __init__(self):
        self.system_prompt = "You are OpenGPT 4o, an exceptionally capable and versatile AI assistant made by linpingta. Your task is to fulfill users query in best possible way"
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def set_model(self, model_name):
        if model_name != self.model_name:
            self.model_name = model_name
            self.load_model()

    def generate_response(self, question, prompt):
        # Combine system prompt, user prompt, and question into a single input
        input_text = f"{self.system_prompt}\nPrompt: {prompt}\nQuestion: {question}"

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Generate a response using the model
        output = self.model.generate(**inputs, max_length=512, do_sample=True, top_p=0.95, temperature=0.7)

        # Decode the generated tokens to text
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response
