"""Ingest PDF/HTML/MD/TXT into Chroma.  python -m rag.ingest data/ [--chunk-size N --overlap N]

Idempotency: chunk id = sha1(source|page|idx|text). Before adding a file we delete all
vectors whose metadata.source == that file, then upsert. Re-running on the same corpus
leaves the count unchanged; editing a file replaces (not duplicates) its chunks.
"""
import argparse, hashlib, re, pathlib
from . import config as C
from .store import collection, embed, dim

def load_file(p: pathlib.Path):
    """-> list of (text, page) ; page is None for non-PDF."""
    ext = p.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(pg.extract_text() or "", i + 1) for i, pg in enumerate(PdfReader(str(p)).pages)]
    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(p.read_text(errors="ignore"), "html.parser")
        for t in soup(["script", "style", "nav", "footer"]): t.decompose()
        return [(soup.get_text(" "), None)]
    if ext in (".md", ".txt"):
        return [(p.read_text(errors="ignore"), None)]
    return []

def chunk_text(text, size, overlap):
    t = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(t):
        j = min(i + size, len(t))
        if j < len(t):                      # prefer to cut on a space
            sp = t.rfind(" ", i + size // 2, j)
            if sp != -1: j = sp
        out.append(t[i:j].strip())
        if j >= len(t): break
        i = max(j - overlap, i + 1)
    return [c for c in out if len(c) > 20]

def ingest_path(root, size=None, overlap=None):
    size, overlap = size or C.CHUNK_SIZE, overlap if overlap is not None else C.CHUNK_OVERLAP
    root = pathlib.Path(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    col = collection()
    before = col.count()
    for f in files:
        pages = load_file(f)
        if not pages: continue
        src = str(f.relative_to(root.parent if root.is_file() else root))
        ids, docs, metas = [], [], []
        n = 0
        for text, page in pages:
            for c in chunk_text(text, size, overlap):
                cid = hashlib.sha1(f"{src}|{page}|{n}|{c}".encode()).hexdigest()[:20]
                ids.append(cid); docs.append(c)
                metas.append({"source": src, "doctype": f.suffix.lower().lstrip("."), "page": page or 0, "chunk_idx": n})
                n += 1
        col.delete(where={"source": src})
        for s in range(0, len(ids), 256):
            sl = slice(s, s + 256)
            col.upsert(ids=ids[sl], documents=docs[sl], metadatas=metas[sl], embeddings=embed(docs[sl]))
        print(f"ingested {src}: {len(ids)} chunks")
    after = col.count()
    print(f"vectors before={before} after={after} | model={C.EMBED_MODEL} dim={dim()} chunk={size} overlap={overlap}")
    return after

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--chunk-size", type=int, default=C.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=C.CHUNK_OVERLAP)
    a = ap.parse_args()
    ingest_path(a.path, a.chunk_size, a.overlap)
