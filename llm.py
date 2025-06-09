import openai
import base64

openai.api_key = "APi key inset here "

def ask_multimodal_llm(text, image_path):
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode('utf-8')

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}
        ]
    )
    return response.choices[0].message["content"]