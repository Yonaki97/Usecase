import ollama

MODEL = "qwen3.5:4b"

response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Balas tepat: Ollama Qwen siap."}],
    think=False,
    stream=False,
)

print("Response:", response["message"]["content"])
