# GenAI Text Summarizer
A simple Generative AI Text Summarizer web app built with Streamlit.

Features:
- Upload text (.txt) or PDF files.
- Paste text directly.
- Choose summarization backend: Hugging Face Transformer (local) or OpenAI (API).
- Short / Medium / Long summary length options.

## Files
- `app.py` - Streamlit frontend
- `summarizer.py` - Summarization wrappers (Hugging Face + OpenAI)
- `utils.py` - Helper functions for file parsing and chunking
- `requirements.txt` - Python dependencies
- `Dockerfile` - Optional containerization
- `.gitignore`
- `LICENSE` - MIT

## Quick start (local)
1. Clone this repo.
2. (Optional) Create and activate a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. To use OpenAI backend, set `OPENAI_API_KEY` environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Running Hugging Face models locally requires GPU for speed and significant RAM for large models.
- If you prefer a lightweight option, use the OpenAI backend (API-based).

## License
MIT
