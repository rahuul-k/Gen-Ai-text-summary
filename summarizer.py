from typing import List, Optional
import os

# Hugging Face
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

class HFSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        if not HF_AVAILABLE:
            raise RuntimeError("Transformers not available. Install `transformers` and `torch`.")
        self.model_name = model_name
        # lazy load
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = pipeline('summarization', model=self.model_name, truncation=True)
        return self._pipe

    def summarize(self, texts: List[str], max_length=150, min_length=40):
        results = []
        for t in texts:
            res = self.pipe(t, max_length=max_length, min_length=min_length, do_sample=False)
            results.append(res[0]['summary_text'])
        return "\n\n".join(results)

class OpenAISummarizer:
    def __init__(self, model: str = 'gpt-4o-mini' ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not found")
        self.model = model
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('Set OPENAI_API_KEY environment variable to use OpenAI backend')
        openai.api_key = key

    def summarize(self, texts: List[str], length: str = 'short'):
        prompt_len = {
            'short': 'Summarize the following text in 2-3 concise sentences.',
            'medium': 'Summarize the following text in 5-6 sentences.',
            'long': 'Summarize the following text in a detailed paragraph.'
        }
        instruction = prompt_len.get(length, prompt_len['short'])
        joined = "\n\n".join(texts)
        system = "You are a helpful summarization assistant."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction + "\n\n" + joined}
        ]
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return resp['choices'][0]['message']['content'].strip()

def get_hf_summarizer(model_name: str = "facebook/bart-large-cnn"):
    return HFSummarizer(model_name=model_name)

def get_openai_summarizer(model: str = 'gpt-4o-mini'):
    return OpenAISummarizer(model=model)
