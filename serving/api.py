"""
API FastAPI — Serving de l'Assistant Juridique RGPD
=====================================================
ENDPOINTS :
    POST /ask          -> Question RGPD -> reponse + sources + confiance
    GET  /health       -> Statut du service
    GET  /model/info   -> Details modele
    POST /feedback     -> Feedback utilisateur

DEMARRAGE :
    uvicorn serving.api:app --reload --port 8000

DOCKER :
    docker-compose up --build

DOCUMENTATION :
    http://localhost:8000/docs
"""

import os, sys, json, time, logging, re
from fastapi.responses import FileResponse
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# APPLICATION
# ================================================================
app = FastAPI(
    title="Assistant Juridique RGPD",
    description="API RAG pour repondre aux questions RGPD/CNIL.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ================================================================
# SCHEMAS
# ================================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)

class QueryResponse(BaseModel):
    question:   str
    answer:     str
    sources:    List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: int

class HealthResponse(BaseModel):
    status:         str
    model:          str
    faiss_vectors:  int
    num_chunks:     int
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    model_name:      str
    lora_version:    str
    last_finetuned:  str
    index_size:      int
    embedding_model: str

class FeedbackRequest(BaseModel):
    question: str
    answer:   str
    rating:   int = Field(..., ge=1, le=5)
    comment:  Optional[str] = None

class FeedbackResponse(BaseModel):
    status:  str
    message: str

# ================================================================
# CHARGEMENT RESSOURCES
# ================================================================
startup_time = time.time()
logger.info("Chargement des ressources RAG...")

from data_preparation.chunking import load_chunks
CHUNKS = load_chunks()
logger.info(f"  -> {len(CHUNKS)} chunks charges")

from rag.retriever import load_index
FAISS_INDEX = load_index()
logger.info(f"  -> Index FAISS : {FAISS_INDEX.ntotal} vecteurs")

from rag.embeddings import get_model as get_embedding_model
get_embedding_model()
logger.info("  -> Modele embeddings pre-charge")

FAST_MODE  = os.environ.get("FAST_MODE", "1") == "1"
MODEL_NAME = os.environ.get("LLM_MODEL",
    "google/flan-t5-small" if FAST_MODE else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
logger.info(f"  -> Chargement LLM : {MODEL_NAME} (fast_mode={FAST_MODE})")

IS_SEQ2SEQ     = "t5" in MODEL_NAME.lower()
_llm_tokenizer = None
_llm_model     = None
_llm_pipeline  = None

if IS_SEQ2SEQ:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _llm_model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
else:
    from transformers import pipeline as hf_pipeline
    _llm_pipeline = hf_pipeline("text-generation", model=MODEL_NAME, device_map="auto")

logger.info("  -> LLM pret")

DEPLOY_CONFIG = {}
if os.path.exists("artifacts/deploy_config.json"):
    with open("artifacts/deploy_config.json") as f:
        DEPLOY_CONFIG = json.load(f)

logger.info("API prete !")

# ================================================================
# RETRIEVAL DEPUIS LE PDF — tout vient des chunks, pas de FAQ codee en dur
# ================================================================

RGPD_KEYWORDS = [
    "rgpd", "dpo", "cnil", "donnee", "donnees", "personnelle", "personnelles",
    "consentement", "traitement", "privacy", "vie privee", "effacement", "oubli",
    "protection", "responsable", "sous-traitant", "transfert", "aipd",
    "violation", "cookies", "droits", "transparence", "minimisation", "registre",
    "article", "sanction", "amende", "notification", "registre", "finalite"
]

def is_rgpd_relevant(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in RGPD_KEYWORDS)

def expand_query(question: str) -> str:
    """
    Query expansion : enrichit la question pour mieux cibler les chunks FAISS.
    """
    q = question.lower()

    # Article specifique : chercher l'article exact dans le PDF
    art_match = re.search(r"article\s+(\d+)", q)
    if art_match:
        num = art_match.group(1)
        return f"Article {num}"   # recherche exacte par titre d'article

    # AIPD / analyse d'impact
    if "aipd" in q or "analyse d'impact" in q or "dpia" in q:
        return "analyse impact protection données AIPD DPIA article 35 risque élevé"
    # Définition générale du RGPD -> Article 1 objet
    if any(p in q for p in ["qu'est-ce", "c'est quoi", "definition", "definit", "signifie", "c est quoi"]) and "article" not in q:
        return "objet règlement données personnelles protection personnes physiques article 1"
    # Droits
    if "droit" in q:
        return question + " accès rectification effacement portabilité opposition article 15 16 17 20 21"
    # Obligations responsable
    if "obligation" in q or ("responsable" in q and "traitement" in q):
        return question + " obligation responsable traitement conformité registre"
    # Consentement
    if "consentement" in q:
        return question + " consentement libre éclairé spécifique base légale article 7"
    # Violation
    if "violation" in q or "fuite" in q or "incident" in q:
        return question + " violation notification 72h CNIL article 33 34"
    # Sanctions
    if "sanction" in q or "amende" in q:
        return question + " sanction amende administrative article 83 20 millions"
    # DPO
    if "dpo" in q or "délégué" in q:
        return question + " délégué protection données DPO article 37 38 39"
    # CNIL
    if "cnil" in q:
        return question + " autorité contrôle CNIL compétence missions pouvoirs article 55"
    return question


def find_article_chunks(article_num: str) -> List[str]:
    """
    Recherche l'Article N du RGPD dans les chunks PDF.

    Distinctions clés :
    - Vrai titre d'article : 'Article 5 Principes' (mot suivant = majuscule)
    - Référence croisée   : 'à l'article 5 du traité' (mot suivant = minuscule)

    Pour l'article 1, le PDF officiel utilise 'Article premier'.
    """
    if article_num == "1":
        # Article 1 = 'Article premier' dans le PDF RGPD officiel
        heading_pat = re.compile(r"\bArticle\s+premier\b", re.IGNORECASE)
        xref_pat    = None
    else:
        # Vrai titre d'article : suivi d'une majuscule (titre) pas d'une minuscule (ref croisée)
        heading_pat = re.compile(
            rf"\bArticle\s+{article_num}\s+[A-ZÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ]", re.UNICODE
        )
        # Ref croisée : précédé de "l'", "de l'", "à l'", "en vertu de l'"
        xref_pat    = re.compile(
            rf"(?:l'|de l'|à l'|en vertu de l')article\s+{article_num}\b", re.IGNORECASE
        )

    priority   = []   # chunk contient le vrai titre d'article (trimmé à partir du titre)
    secondary  = []   # chunk mentionne l'article sans titre explicite

    for chunk in CHUNKS:
        m = heading_pat.search(chunk)
        if m:
            # Trimmer le chunk pour commencer exactement au titre de l'article
            priority.append(chunk[m.start():])
        elif xref_pat and xref_pat.search(chunk):
            secondary.append(chunk)

    # Préférer les vrais titres d'articles ; ne tomber sur les cross-refs qu'en dernier recours
    return (priority if priority else secondary)[:5]


def retrieve_chunks(question: str) -> tuple:
    """
    Retourne (chunks, scores) depuis le PDF — tout vient des chunks du RGPD.

    Stratégie :
    1. Article explicite ("article 5")      -> recherche textuelle "Article 5" dans chunks
    2. "Qu'est-ce que X" / "C'est quoi X"  -> trouve l'article du PDF qui définit X
    3. Autre question                       -> FAISS sémantique + query expansion
    """
    from rag.retriever import search_with_scores

    q = question.lower()

    # --- Stratégie 1 : article explicitement cité ---
    art_match = re.search(r"article\s+(\d+)", q)
    if art_match:
        chunks = find_article_chunks(art_match.group(1))
        if chunks:
            return chunks, [0.93] * len(chunks)

    # --- Stratégie 2 : questions définitionnelles -> bon article dans le PDF ---
    definitional = any(p in q for p in [
        "qu'est-ce", "c'est quoi", "définit", "signifie", "qu est ce",
        "c est quoi", "kesako", "explain", "explain me"
    ])
    if definitional:
        # Carte question -> article du PDF qui répond
        definitional_map = [
            (["rgpd", "règlement", "reglement"],        "1"),   # Art 1 : Objet (= Article premier)
            (["dpo", "délégué", "delegue"],              "37"),  # Art 37 : DPO
            (["consentement"],                           "7"),   # Art 7 : Consentement
            (["aipd", "analyse d'impact", "dpia"],       "35"),  # Art 35 : AIPD
            (["violation", "fuite", "incident"],         "33"),  # Art 33 : Notification
            (["effacement", "oubli"],                    "17"),  # Art 17 : Effacement
            (["portabilité", "portabilite"],             "20"),  # Art 20 : Portabilité
            (["sous-traitant", "sous traitant"],         "28"),  # Art 28 : Sous-traitant
            (["registre"],                               "30"),  # Art 30 : Registre
        ]
        for keywords, art_num in definitional_map:
            if any(k in q for k in keywords):
                art_chunks = find_article_chunks(art_num)
                if art_num == "1":
                    # Pour RGPD : chunk 0 = en-tête du règlement (définition claire)
                    # + chunk "Article premier Objet et objectifs"
                    chunks = []
                    if CHUNKS:
                        chunks.append(CHUNKS[0])
                    chunks += [c for c in art_chunks if c != CHUNKS[0]]
                else:
                    chunks = art_chunks
                if chunks:
                    return chunks[:3], [0.92] * 3

    # --- Stratégie 3 : FAISS sémantique ---
    expanded = expand_query(question)
    results  = search_with_scores(expanded, FAISS_INDEX, CHUNKS, k=5)
    chunks   = [c for c, _ in results]
    scores   = [s for _, s in results]
    return chunks, scores


def compute_confidence(scores: List[float]) -> float:
    if not scores:
        return 0.0
    top     = sorted(scores, reverse=True)[:3]
    weights = [0.5, 0.3, 0.2]
    conf    = sum(s * w for s, w in zip(top, weights[:len(top)]))
    return round(min(float(conf), 1.0), 3)


def _extractive_answer(question: str, context_chunks: List[str]) -> str:
    """
    Reponse extractive depuis les chunks PDF.
    Re-ranking par chevauchement de mots-cles, puis extraction propre.
    """
    stop = {
        'le','la','les','de','du','des','un','une','est','en','et','ou',
        'que','qui','ce','se','sa','son','ses','ne','pas','plus','pour',
        'dans','avec','par','sur','au','aux','il','elle','ils','elles',
        'a','d','l','m','s','n','je','tu','nous','vous','mon','ton',
        'leur','leurs','y','si','car','mais','donc','ni','qu','c'
    }
    q_words = {w for w in question.lower().split() if w not in stop and len(w) > 2}

    best_chunk = context_chunks[0]
    best_score = -1
    for chunk in context_chunks:
        cl    = chunk.lower()
        score = sum(1 for w in q_words if w in cl)
        # À score égal, préférer le chunk le plus long (plus informatif)
        if score > best_score or (score == best_score and len(chunk) > len(best_chunk)):
            best_score = score
            best_chunk = chunk

    text = best_chunk.replace('\n', ' ').strip()

    # Sauter les préambules techniques (date, Journal officiel, etc.)
    # vers le vrai contenu : soit "RÈGLEMENT (UE)", soit "Article Premier/N"
    content_start = re.search(
        r'\bRÈGLEMENT\s+\(UE\)|\bArticle\s+(?:premier|\d+)\b',
        text, re.IGNORECASE
    )
    if content_start and content_start.start() > 80:
        text = text[content_start.start():]

    if len(text) > 700:
        cut       = text[:700]
        last_stop = max(cut.rfind('. '), cut.rfind('! '), cut.rfind('? '))
        text      = cut[:last_stop + 1] if last_stop > 150 else cut + '…'
    return text


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    Genere une reponse depuis les chunks PDF.
    - IS_SEQ2SEQ (flan-t5) : reponse extractive (toujours en francais)
    - TinyLlama             : generation causal LM
    """
    from rag.prompt_templates import build_no_context_response

    if not context_chunks:
        return "Je n'ai pas trouvé d'information correspondante dans le texte du RGPD. Essayez de reformuler votre question."

    if IS_SEQ2SEQ:
        return _extractive_answer(question, context_chunks)

    # TinyLlama
    from rag.prompt_templates import build_rag_prompt
    prompt = build_rag_prompt(question, context_chunks)
    try:
        result = _llm_pipeline(prompt, max_new_tokens=150, do_sample=False, repetition_penalty=1.15)
        output = result[0].get("generated_text", "")
        if "REPONSE :" in output:
            answer = output.split("REPONSE :")[-1].strip()
        else:
            answer = output[len(prompt):].strip()
        for stop_token in ["QUESTION :", "CONTEXTE :", "---", "<|user|>"]:
            answer = answer.split(stop_token)[0].strip()
        forbidden = ["je pense", "probablement", "peut-etre", "selon moi", "je crois"]
        if any(f in answer.lower() for f in forbidden):
            return build_no_context_response()
        return answer if len(answer) >= 10 else build_no_context_response()
    except Exception as e:
        logger.error(f"Erreur generation : {e}")
        return build_no_context_response()


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/")
async def root():
    """Sert l'interface HTML du chatbot."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "Assistant RGPD actif", "docs": "/docs"}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Pose une question RGPD -> reponse extraite du PDF + sources + confiance."""
    t0 = time.time()
    logger.info(f"[/ask] Question : {request.question}")

    if not is_rgpd_relevant(request.question):
        return QueryResponse(
            question   = request.question,
            answer     = "Cette question ne semble pas concerner le RGPD. Veuillez reformuler.",
            sources    = [],
            confidence = 0.0,
            latency_ms = int((time.time() - t0) * 1000)
        )

    # Retrieval depuis le PDF (article search ou FAISS)
    context_chunks, scores = retrieve_chunks(request.question)
    logger.info(f"  -> {len(context_chunks)} chunks (scores: {[round(s,3) for s in scores]})")

    answer     = generate_answer(request.question, context_chunks)
    confidence = compute_confidence(scores)
    latency_ms = int((time.time() - t0) * 1000)

    logger.info(f"  -> Confiance={confidence} | Latence={latency_ms}ms")
    return QueryResponse(
        question   = request.question,
        answer     = answer,
        sources    = context_chunks[:3],
        confidence = confidence,
        latency_ms = latency_ms
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status         = "ok",
        model          = MODEL_NAME,
        faiss_vectors  = FAISS_INDEX.ntotal,
        num_chunks     = len(CHUNKS),
        uptime_seconds = round(time.time() - startup_time, 1)
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_name      = MODEL_NAME,
        lora_version    = DEPLOY_CONFIG.get("model_version", "none"),
        last_finetuned  = DEPLOY_CONFIG.get("deployed_at", "N/A"),
        index_size      = FAISS_INDEX.ntotal,
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Enregistre un feedback utilisateur dans artifacts/feedback.jsonl."""
    os.makedirs("artifacts", exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question":  request.question,
        "answer":    request.answer,
        "rating":    request.rating,
        "comment":   request.comment or ""
    }
    with open("artifacts/feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return FeedbackResponse(status="ok", message="Feedback enregistre.")
