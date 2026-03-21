# PathForge

🚀 **AI-Adaptive Skill Pathway Generator** 
**Frontend Live URL:** https://aionboardingpathforge.netlify.app  
**Backend Live URL:** https://pathforge-backend-q64h.onrender.com

PathForge leverages advanced NLP, embedding models, and the speed of Groq's LLaMA 3.3 to create highly personalized, adaptive learning and career pathways based on specific job roles and experience levels. It moves beyond static roadmaps by synthesizing curriculum data with state-of-the-art LLM reasoning.

---

## ✨ Core Features & Functionality

PathForge is designed to answer: **"Given my role and experience, what do I need to learn next?"**

1.  **Adaptive Pathway Generation (`/api/analyze`):**
    *   Takes a `role` (e.g., "Software Engineer") and an `experience_level` (e.g., "junior").
    *   Uses context from our skill/course databases and reasoning from the Groq LLM to construct a multi-stage learning pathway, including recommended courses, core skills, and nice-to-have skills.
2.  **LLM Reasoning & Chat (`/api/chat`):**
    *   Provides a conversational interface grounded in the application's knowledge base (skills, courses, O\*NET data).
3.  **Data Grounding:**
    *   The intelligence is anchored by rich, structured data files (`role_competencies.json`, `course_catalog.json`, `onet_skills.csv`) ensuring recommendations are relevant and objective.
4.  **Model Warming:**
    *   The application pre-loads both the Sentence Transformer model and initializes the Groq client at startup, ensuring fast initial response times.

---

## 🛠️ Technology Stack Deep Dive

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Web Framework** | FastAPI, Uvicorn | High-performance API layer, asynchronous support. |
| **LLM Engine** | Groq SDK | Utilizes LLaMA 3.3 for fast, complex reasoning and pathway construction. |
| **Vector Search** | SentenceTransformers | Used for efficient semantic matching of skills/concepts. |
| **NLP/Parsing** | spaCy, pdfplumber, python-docx | Processing structured and unstructured text data (if files were used for input). |
| **Data Science** | Pandas, NumPy, Scikit-learn | Data manipulation, evaluation, and potential future metrics. |
| **Database** | SQLAlchemy + aiosqlite | Lightweight SQLite management for session/state persistence. |
| **Deployment** | Docker, Render, Netlify | Containerization for consistent builds, cloud hosting for production scale. |

---

## 🚀 Getting Started (Setup)

### Prerequisites
1.  Python 3.10+
2.  A **Groq API Key** (required for LLM inference).

### 1. Clone and Environment Setup

```bash
git clone <your-repo-url>
cd <repo-directory>
python -m venv .venv
source .venv/bin/activate   # macOS/Linux | .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Environment Variables
You **must** set your Groq key:
```bash
export GROQ_API_KEY=your_secret_groq_key_here
export DEBUG=false
```

### 3. Run Locally
```bash
# Start the FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The application will automatically initialize the database, warm the embedding model, and connect to Groq.

---

## 🌐 Deployment Notes (Render)

The Dockerfile ensures a self-contained environment. Key considerations for Render:

1.  **Port Binding:** The `CMD` in the Dockerfile correctly listens to the `$PORT` variable provided by Render:
    ```dockerfile
    # Inside Dockerfile CMD:
    CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
    ```
2.  **CORS Configuration:** Cross-Origin Resource Sharing is configured in `backend/app/main.py` to explicitly allow traffic from your Netlify domain:
    ```python
    allow_origins=[
        "https://aionboardingpathforge.netlify.app",
        # ... other origins
    ],
    ```

---

## API Reference

All endpoints require `Content-Type: multipart/form-data` for analysis endpoints.

| Endpoint | Method | Required Input | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | Returns `{ "status": "ok" }`. |
| `/docs` | `GET` | None | Interactive API documentation (Swagger UI). |
| `/api/analyze` | `POST` | `role`, `experience_level` | Generates the structured learning pathway. |
| `/api/analyze/stream` | `POST` | `role`, `experience_level` | Streams the pathway generation response chunks. |
| `/api/chat` | `POST` | `messages` | General chat interaction with the LLM. |
| `/api/quiz` | `POST` | Varies | Endpoints for retrieving or validating quizzes. |

---

## Next Steps & Enhancements
*   Implement dynamic difficulty adjustment based on quiz performance.
*   Integrate external API calls for up-to-date course data, replacing hardcoded JSON.
*   Add user persistence via the SQLite database for tracking completed modules.

---

