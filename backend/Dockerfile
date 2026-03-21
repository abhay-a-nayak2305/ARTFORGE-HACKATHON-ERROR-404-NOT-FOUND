FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# FIX: WORKDIR is /app. The render.yaml sets dockerContext: ./backend,
# so Docker only sees the contents of the backend/ folder.
# That means COPY . . copies backend/* into /app directly (no /app/backend subdir).
# The old CMD did "cd /app/backend" which doesn't exist and crashes the server.
WORKDIR /app

# Install CPU-only PyTorch first (largest download, cached separately)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Create requirements.txt inline ───────────────────────
RUN cat > /app/requirements.txt << 'REQS'
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pydantic==2.7.1
pydantic-settings==2.3.1
groq==0.9.0
transformers==4.41.2
sentence-transformers==3.0.1
spacy==3.7.5
pdfplumber==0.11.0
python-docx==1.1.2
numpy==1.26.4
pandas==2.2.2
networkx==3.3
scikit-learn==1.5.0
sqlalchemy==2.0.30
aiosqlite==0.20.0
httpx==0.27.0
aiofiles==23.2.1
structlog==24.2.0
REQS

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN python -m spacy download en_core_web_sm

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy application code ─────────────────────────────────
# Since dockerContext is ./backend, this copies backend/* → /app/*
COPY . .

# ── Create data directories ───────────────────────────────
RUN mkdir -p /app/data /app/uploads

RUN python3 << 'PYEOF'
import json, os

# ── onet_skills.csv ───────────────────────────────────────
onet_csv = """onet_code,title,category,description
15-1251.00,Python,Programming Languages,Python programming language
15-1251.01,JavaScript,Programming Languages,JavaScript programming language
15-1251.02,TypeScript,Programming Languages,TypeScript programming language
15-1251.03,Java,Programming Languages,Java programming language
15-1251.04,Go,Programming Languages,Go programming language
15-1252.00,React,Frontend Frameworks,React JavaScript framework
15-1252.01,Vue,Frontend Frameworks,Vue.js JavaScript framework
15-1252.02,Angular,Frontend Frameworks,Angular TypeScript framework
15-1252.03,Node.js,Backend Frameworks,Node.js runtime environment
15-1252.04,Django,Backend Frameworks,Django Python web framework
15-1252.05,FastAPI,Backend Frameworks,FastAPI Python web framework
15-1252.06,Flask,Backend Frameworks,Flask Python micro framework
15-1253.00,Docker,DevOps Tools,Docker containerisation platform
15-1253.01,Kubernetes,DevOps Tools,Kubernetes container orchestration
15-1253.02,Terraform,DevOps Tools,Terraform infrastructure as code
15-1253.03,Ansible,DevOps Tools,Ansible automation tool
15-1253.04,GitHub Actions,DevOps Tools,GitHub Actions CI/CD platform
15-1254.00,AWS,Cloud Platforms,Amazon Web Services cloud platform
15-1254.01,GCP,Cloud Platforms,Google Cloud Platform
15-1254.02,Azure,Cloud Platforms,Microsoft Azure cloud platform
15-1255.00,PostgreSQL,Databases,PostgreSQL relational database
15-1255.01,MySQL,Databases,MySQL relational database
15-1255.02,MongoDB,Databases,MongoDB NoSQL database
15-1255.03,Redis,Databases,Redis in-memory data store
15-1255.04,SQL,Databases,Structured Query Language
15-1256.00,TensorFlow,ML Frameworks,TensorFlow machine learning framework
15-1256.01,PyTorch,ML Frameworks,PyTorch deep learning framework
15-1256.02,Scikit-learn,ML Frameworks,Scikit-learn machine learning library
15-1256.03,Pandas,Data Science,Pandas data manipulation library
15-1256.04,NumPy,Data Science,NumPy numerical computing library
15-1256.05,Spark,Big Data,Apache Spark distributed computing
15-1256.06,Machine Learning,Data Science,Machine learning concepts
15-1256.07,Deep Learning,Data Science,Deep learning and neural networks
15-1256.08,NLP,Data Science,Natural Language Processing
15-1256.09,MLOps,Data Science,Machine learning operations
15-1256.10,Feature Engineering,Data Science,Feature engineering techniques
15-1257.00,REST APIs,Architecture,RESTful API design and development
15-1257.01,GraphQL,Architecture,GraphQL query language
15-1257.02,Microservices,Architecture,Microservices architecture pattern
15-1257.03,System Design,Architecture,System design and architecture
15-1257.04,CI/CD,DevOps Practices,Continuous integration and deployment
15-1257.05,Git,Version Control,Git version control system
15-1257.06,Linux,Operating Systems,Linux operating system administration
15-1258.00,Statistics,Mathematics,Statistical analysis and probability
15-1258.01,A/B Testing,Analytics,A/B testing and experimentation
15-1259.00,Product Strategy,Product Management,Product strategy and vision
15-1259.01,Agile,Methodologies,Agile project management methodology
15-1259.02,Scrum,Methodologies,Scrum framework for agile development
15-1259.03,OKRs,Product Management,Objectives and key results framework
15-1259.04,Roadmapping,Product Management,Product roadmap planning
15-1259.05,Stakeholder Management,Soft Skills,Managing stakeholder relationships
15-1259.06,User Research,UX Design,User research and usability testing
15-1259.07,JIRA,Project Management,JIRA project tracking tool
15-1259.08,Figma,Design Tools,Figma UI/UX design tool
15-1260.00,Communication,Soft Skills,Professional communication skills
15-1260.01,Leadership,Soft Skills,Leadership and team management
15-1260.02,Problem Solving,Soft Skills,Analytical problem solving
15-1260.03,Testing,Software Engineering,Software testing and QA
15-1260.04,Code Review,Software Engineering,Code review best practices
"""

with open("/app/data/onet_skills.csv", "w") as f:
    f.write(onet_csv.strip())

# ── course_catalog.json ───────────────────────────────────
course_catalog = {
    "Python": {"provider": "Official Python Docs", "url": "https://docs.python.org/3/tutorial/", "duration_days": 4, "level": "beginner"},
    "JavaScript": {"provider": "MDN Web Docs", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", "duration_days": 4, "level": "beginner"},
    "TypeScript": {"provider": "TypeScript Docs", "url": "https://www.typescriptlang.org/docs/", "duration_days": 3, "level": "intermediate"},
    "React": {"provider": "React Official Docs", "url": "https://react.dev/learn", "duration_days": 5, "level": "intermediate"},
    "Docker": {"provider": "Docker Docs", "url": "https://docs.docker.com/get-started/", "duration_days": 3, "level": "intermediate"},
    "Kubernetes": {"provider": "Kubernetes Docs", "url": "https://kubernetes.io/docs/tutorials/", "duration_days": 5, "level": "advanced"},
    "AWS": {"provider": "AWS Skill Builder", "url": "https://skillbuilder.aws/", "duration_days": 6, "level": "intermediate"},
    "Machine Learning": {"provider": "fast.ai", "url": "https://www.fast.ai/", "duration_days": 7, "level": "intermediate"},
    "System Design": {"provider": "System Design Primer", "url": "https://github.com/donnemartin/system-design-primer", "duration_days": 5, "level": "advanced"},
    "PostgreSQL": {"provider": "PostgreSQL Tutorial", "url": "https://www.postgresql.org/docs/current/tutorial.html", "duration_days": 3, "level": "intermediate"},
    "TensorFlow": {"provider": "TensorFlow Tutorials", "url": "https://www.tensorflow.org/tutorials", "duration_days": 6, "level": "advanced"},
    "PyTorch": {"provider": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "duration_days": 6, "level": "advanced"},
    "REST APIs": {"provider": "REST API Tutorial", "url": "https://restfulapi.net/", "duration_days": 2, "level": "intermediate"},
    "Git": {"provider": "Pro Git Book", "url": "https://git-scm.com/book/en/v2", "duration_days": 2, "level": "beginner"},
    "Terraform": {"provider": "HashiCorp Learn", "url": "https://developer.hashicorp.com/terraform/tutorials", "duration_days": 4, "level": "advanced"},
    "SQL": {"provider": "SQLZoo", "url": "https://sqlzoo.net/", "duration_days": 3, "level": "beginner"},
    "Deep Learning": {"provider": "deeplearning.ai", "url": "https://www.deeplearning.ai/courses/", "duration_days": 8, "level": "advanced"},
    "Agile": {"provider": "Agile Alliance", "url": "https://www.agilealliance.org/agile101/", "duration_days": 2, "level": "beginner"},
    "Figma": {"provider": "Figma Learn", "url": "https://www.figma.com/resources/learn-design/", "duration_days": 3, "level": "beginner"},
    "Statistics": {"provider": "Khan Academy", "url": "https://www.khanacademy.org/math/statistics-probability", "duration_days": 5, "level": "beginner"},
    "Pandas": {"provider": "Pandas Docs", "url": "https://pandas.pydata.org/docs/getting_started/", "duration_days": 3, "level": "beginner"},
    "Linux": {"provider": "Linux Journey", "url": "https://linuxjourney.com/", "duration_days": 4, "level": "intermediate"},
    "MLOps": {"provider": "MLOps Community", "url": "https://mlops.community/", "duration_days": 5, "level": "advanced"},
    "NLP": {"provider": "Hugging Face Course", "url": "https://huggingface.co/learn/nlp-course/", "duration_days": 6, "level": "advanced"},
}

with open("/app/data/course_catalog.json", "w") as f:
    json.dump(course_catalog, f, indent=2)

# ── role_competencies.json ────────────────────────────────
role_competencies = {
    "Software Engineer": {
        "onet_soc": "15-1252.00",
        "skills": ["Python","JavaScript","TypeScript","System Design","REST APIs",
                   "Git","Docker","SQL","Testing","CI/CD","Code Review","Microservices","Linux","AWS"],
        "core_skills": ["Python","System Design","REST APIs","Git"],
        "nice_to_have": ["Kubernetes","Terraform","GraphQL"]
    },
    "Data Scientist": {
        "onet_soc": "15-2051.00",
        "skills": ["Python","Machine Learning","Statistics","Deep Learning","Pandas",
                   "NumPy","SQL","TensorFlow","PyTorch","Feature Engineering","MLOps","A/B Testing","Spark","NLP"],
        "core_skills": ["Python","Machine Learning","Statistics","SQL"],
        "nice_to_have": ["Spark","Kubernetes","Tableau"]
    },
    "Product Manager": {
        "onet_soc": "11-2021.00",
        "skills": ["Product Strategy","Agile","Scrum","Roadmapping","Stakeholder Management",
                   "OKRs","User Research","A/B Testing","JIRA","Communication","Leadership"],
        "core_skills": ["Product Strategy","Agile","Stakeholder Management","OKRs"],
        "nice_to_have": ["SQL","Figma","Python"]
    },
    "DevOps Engineer": {
        "onet_soc": "15-1244.00",
        "skills": ["Docker","Kubernetes","Terraform","Ansible","AWS","CI/CD","GitHub Actions","Linux","Git","Python"],
        "core_skills": ["Docker","Kubernetes","CI/CD","Linux"],
        "nice_to_have": ["GCP","Azure","Rust"]
    },
}

with open("/app/data/role_competencies.json", "w") as f:
    json.dump(role_competencies, f, indent=2)

print("Data files created successfully.")
PYEOF

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# FIX: The dockerContext in render.yaml is ./backend, so Docker copies
# the contents of backend/ directly into /app (not into /app/backend).
# The app module is therefore at /app/app/main.py → "app.main:app".
# No "cd /app/backend" needed — just run uvicorn from /app directly.
CMD ["sh", "-c", "cd /app/backend && PYTHONPATH=/app/backend uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
