"""Fake News Detection using Machine Learning - FastAPI backend.
Run:  python main.py   (the address is printed and the browser opens automatically)"""
import os, re, json, random, socket, sqlite3, threading, webbrowser
from datetime import datetime
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.join(os.path.dirname(BASE), "frontend")
DB = os.path.join(BASE, "news.db")

# ---------- Built-in sample dataset (1 = fake, 0 = real) ----------
TOPICS = ["the new vaccine", "the election results", "the stock market", "the latest smartphone", "climate change",
          "school exams", "the cricket team", "petrol prices", "the city council", "a famous actor",
          "the health ministry", "the central bank", "the railway project", "online banking"]
FAKE = ["SHOCKING: {t} secret EXPOSED, they don't want you to know the truth!!!",
        "You won't believe what happened to {t}, share before it gets deleted!",
        "BREAKING!!! {t} is a total hoax and everyone is being lied to, wake up!",
        "Miracle discovery about {t} that doctors hate, 100% proven and guaranteed!",
        "Insider reveals {t} is a massive government cover-up, the media is hiding it!",
        "URGENT: {t} will destroy everything by tomorrow, forward to all your friends!",
        "Unbelievable! {t} caught in a huge scandal, what they did will SHOCK you",
        "Anonymous source claims {t} was faked, the real truth finally leaked online!!!",
        "Scientists are speechless: {t} cures everything overnight, no one is talking about it",
        "Viral post says {t} is banned worldwide, click now before it disappears"]
REAL = ["According to officials, {t} saw a modest change this week, a report said.",
        "The ministry said on Tuesday that {t} will be reviewed after a detailed study.",
        "Researchers published a peer-reviewed study on {t} in a scientific journal.",
        "Police told reporters that an investigation into {t} is ongoing and no arrests have been made.",
        "Analysts at the university said {t} is expected to grow by two percent next quarter.",
        "The government announced a new policy on {t}, officials confirmed in a statement.",
        "A spokesperson for the council said {t} would be discussed at the next meeting.",
        "Data released by the statistics office shows {t} rose slightly compared with last year.",
        "Experts told the press that {t} requires further evidence before conclusions are drawn.",
        "Local authorities confirmed that {t} was discussed during a public hearing on Monday."]
FS = ["", " Share now!", " Wake up people!", " Don't trust the media!"]
RS = ["", " Officials added more details later.", " The full report is available online.", " No further comment was made."]

def sample_data():
    random.seed(7)
    X, y = [], []
    for tpl in FAKE:
        for t in TOPICS:
            for s in FS: X.append(tpl.format(t=t) + s); y.append(1)
    for tpl in REAL:
        for t in TOPICS:
            for s in RS: X.append(tpl.format(t=t) + s); y.append(0)
    return X, y

# ---------- Model ----------
M = {}

def train(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words='english')
    A = vec.fit_transform(Xtr)
    lr = LogisticRegression(max_iter=2000, C=3).fit(A, ytr)
    nb = MultinomialNB(alpha=0.3).fit(A, ytr)
    B = vec.transform(Xte)
    pl = lr.predict(B)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    M.update(vec=vec, lr=lr, nb=nb, vocab=vec.vocabulary_, coef=lr.coef_[0],
             metrics={"accuracy": round(accuracy_score(yte, pl) * 100, 1), "precision": round(precision_score(yte, pl, zero_division=0) * 100, 1),
                      "recall": round(recall_score(yte, pl, zero_division=0) * 100, 1), "f1": round(f1_score(yte, pl, zero_division=0) * 100, 1),
                      "nb_accuracy": round(accuracy_score(yte, nb.predict(B)) * 100, 1), "samples": len(X)},
             top_fake=[[n, round(float(c), 2)] for n, c in zip(names[order[::-1][:8]], lr.coef_[0][order[::-1][:8]])],
             top_real=[[n, round(float(abs(c)), 2)] for n, c in zip(names[order[:8]], lr.coef_[0][order[:8]])])

train(*sample_data())

# ---------- Database ----------
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

with db() as c:
    c.execute("CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, text TEXT, verdict TEXT, confidence REAL, credibility REAL)")

# ---------- Analysis ----------
CLICK = ["you won't believe", "shocking", "they don't want you to know", "share before", "miracle", "100%", "doctors hate", "exposed", "wake up", "hoax", "cover-up", "leaked", "banned worldwide", "secret"]
ATTR = ["according to", "said", "report", "official", "study", "ministry", "police", "told", "confirmed", "researchers", "data"]

def analyze(text: str):
    text = text.strip()
    v = M["vec"].transform([text])
    p = 0.6 * M["lr"].predict_proba(v)[0][1] + 0.4 * M["nb"].predict_proba(v)[0][1]
    low = text.lower()
    words = re.findall(r"[A-Za-z']+", text)
    flags = []
    caps = [w for w in words if len(w) >= 3 and w.isupper()]
    if len(caps) >= 2: flags.append(f"{len(caps)} ALL-CAPS words (emotional shouting)")
    if text.count("!") >= 2: flags.append("Multiple exclamation marks")
    hit = [c for c in CLICK if c in low]
    if hit: flags.append("Clickbait phrases: " + ", ".join(hit[:3]))
    if not any(a in low for a in ATTR): flags.append("No source or attribution mentioned")
    if len(words) < 6: flags.append("Very short text, low context")
    cred = float(np.clip((1 - p) * 100 - 4 * len(flags) + (6 if not flags else 0), 0, 100))
    conf = max(p, 1 - p) * 100
    verdict = "UNCERTAIN" if conf < 60 else ("FAKE" if p >= 0.5 else "REAL")
    toks = []
    for t in re.findall(r"\w+|\W+", text)[:500]:
        i = M["vocab"].get(t.lower())
        toks.append([t, round(float(M["coef"][i]), 2) if i is not None and re.match(r"\w", t) else 0])
    return {"verdict": verdict, "fake_probability": round(p * 100, 1), "confidence": round(conf, 1), "credibility": round(cred, 1),
            "lr": round(float(M["lr"].predict_proba(v)[0][1]) * 100, 1), "nb": round(float(M["nb"].predict_proba(v)[0][1]) * 100, 1),
            "flags": flags, "tokens": toks}

# ---------- API ----------
class Item(BaseModel):
    text: str = Field(min_length=15, max_length=5000)

class Many(BaseModel):
    texts: List[str]

class Row(BaseModel):
    text: str
    label: int = Field(ge=0, le=1)

class Rows(BaseModel):
    rows: List[Row]

app = FastAPI(title="Fake News Detection using Machine Learning")

@app.post("/api/analyze")
def api_analyze(i: Item, save: bool = True):
    r = analyze(i.text)
    if save:
        with db() as c:
            c.execute("INSERT INTO checks(created,text,verdict,confidence,credibility) VALUES(?,?,?,?,?)",
                      (datetime.now().strftime("%Y-%m-%d %H:%M"), i.text[:300], r["verdict"], r["confidence"], r["credibility"]))
    return r

@app.post("/api/batch")
def api_batch(m: Many):
    out = []
    for t in [x for x in m.texts if len(x.strip()) >= 15][:100]:
        r = analyze(t)
        out.append({"text": t[:120], "verdict": r["verdict"], "confidence": r["confidence"], "credibility": r["credibility"]})
    return {"results": out}

@app.get("/api/stats")
def api_stats():
    with db() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM checks ORDER BY id DESC LIMIT 200")]
    return {"metrics": M["metrics"], "top_fake": M["top_fake"], "top_real": M["top_real"], "history": rows}

@app.delete("/api/history")
def api_clear():
    with db() as c:
        c.execute("DELETE FROM checks")
    return {"ok": True}

@app.post("/api/train")
def api_train(r: Rows):
    if len(r.rows) < 20 or len({x.label for x in r.rows}) < 2:
        raise HTTPException(400, "Need at least 20 rows with both fake (1) and real (0) labels.")
    train([x.text for x in r.rows], [x.label for x in r.rows])
    return {"metrics": M["metrics"]}

app.mount("/static", StaticFiles(directory=FRONT), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONT, "index.html"), headers={"Cache-Control": "no-store"})

@app.get("/login")
def login_redirect():
    return RedirectResponse("/")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

def free_port():
    for p in (8080, 5050, 9090, 7777, 8765, 5000, 3030, 8001, 8888):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError:
            s.close()
    return 0

if __name__ == "__main__":
    import uvicorn
    port = free_port()
    print(f"\n>>> Open in browser: http://127.0.0.1:{port}\n")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
