from openai import OpenAI

def get_openai_client(api_key):
    """Initialize OpenAI client."""
    return OpenAI(api_key=api_key)

def chat_completion(client, prompt, context, history):
    """Generate a chat completion using OpenAI GPT-3.5."""
    messages = [{"role": "system", "content": f"Use the following context: {context}"}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    response = "".join([chunk['choices'][0]['delta']['content'] for chunk in stream])
    return response
