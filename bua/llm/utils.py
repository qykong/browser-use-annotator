from io import BytesIO
import base64
import openai


def image_to_base64(image):
    """Convert PIL Image to base64 string for OpenAI API"""
    if image is None:
        return None
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def call_openai_vlm(prompt, images, model="gpt-4o-mini-vision-preview", api_key=None, base_url=None):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    
    for image in images:
        image_b64 = image_to_base64(image)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_b64}
        })

    response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    
    return response.choices[0].message.content.strip()