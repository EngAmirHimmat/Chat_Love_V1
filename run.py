from transformers import pipeline

chatbot = pipeline("text-generation", model="./trained_chat_model")

response = chatbot("مرحبًا، كيف حالك؟", max_length=100)
print(response[0]["generated_text"])
