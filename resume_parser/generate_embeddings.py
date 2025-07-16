import os
import json
import time
import logging
import tempfile
import argparse
from typing import List, Dict

import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import chromadb

# ——— Configure logging ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ——— Global resources ———
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 5000

chroma_client = chromadb.PersistentClient(path="chroma_db_api")
model = SentenceTransformer(MODEL_NAME)


# ——— Normalization ———
def normalize_skill(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def normalize_generic(s: str) -> str:
    return (
        s.strip()
         .lower()
         .replace("-", " ")
         .replace(".", "")
         .replace(",", "")
         .replace("/", " ")
         .replace("  ", " ")
    )


# ——— Batch uploader ———
def process_in_batches(
    collection,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    documents: List[str],
    batch_size: int = BATCH_SIZE,
) -> None:
    total = len(ids)
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, total)
        logging.info(f"[batch {i+1}/{num_batches}] adding {end-start} items")
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            documents=documents[start:end],
        )


# ——— Core routines ———
def build_embeddings(
    csv_path: str,
    id_col: str,
    text_col: str,
    collection_name: str,
    normalizer,
):
    logging.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    if id_col not in df or text_col not in df:
        raise ValueError(f"CSV must contain '{id_col}' and '{text_col}' columns")

    df[id_col] = df[id_col].astype(int)
    df = df[[id_col, text_col]].drop_duplicates().dropna()
    df[text_col] = df[text_col].astype(str).apply(normalizer)

    texts = df[text_col].tolist()
    ids = df[id_col].astype(str).tolist()

    logging.info(f"Computing {len(texts)} embeddings with {MODEL_NAME}")
    embs = model.encode(texts, normalize_embeddings=True).tolist()

    # (Re)create collection
    try:
        chroma_client.delete_collection(collection_name)
        logging.info(f"Deleted existing '{collection_name}' collection")
    except Exception:
        pass

    col = chroma_client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    process_in_batches(col, ids, embs,
                       metadatas=[{collection_name.rstrip("s"): t} for t in texts],
                       documents=texts)
    logging.info(f"Finished building '{collection_name}' embeddings")


def add_to_collection(
    collection_name: str,
    entries: List[Dict[str, str]],
):
    try:
        col = chroma_client.get_collection(collection_name)
    except ValueError:
        available = [c.name for c in chroma_client.list_collections()]
        raise ValueError(
            f"Collection '{collection_name}' not found. Available: {available}"
        )

    ids, embs, metas, docs = [], [], [], []
    for e in entries:
        if "id" not in e or "text" not in e:
            raise ValueError("Each entry needs 'id' and 'text'")
        txt = e["text"]
        if collection_name == "skills":
            txt = normalize_skill(txt)
            meta = {"skill": txt}
        elif collection_name in ("colleges", "companies", "titles"):
            txt = normalize_generic(txt)
            meta = {collection_name.rstrip("s"): txt}
        else:
            meta = {"text": txt}

        logging.info(f"Encoding single entry '{txt}'")
        emb = model.encode([txt], normalize_embeddings=True)[0].tolist()

        ids.append(str(e["id"]))
        embs.append(emb)
        metas.append(meta)
        docs.append(txt)

    col.add(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
    logging.info(f"Added {len(ids)} entries to '{collection_name}'")


# ——— CLI ———
def main():
    parser = argparse.ArgumentParser(description="Build / update ChromaDB embeddings")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # skills
    p1 = sub.add_parser("skills", help="Build skills embeddings")
    p1.add_argument("csv", help="Path to skills CSV (s_no,name)")

    # colleges
    p2 = sub.add_parser("colleges", help="Build colleges embeddings")
    p2.add_argument("csv", help="Path to colleges CSV (CollegeID,College Name)")

    # companies
    p3 = sub.add_parser("companies", help="Build companies embeddings")
    p3.add_argument("csv", help="Path to companies CSV (Company ID,Company)")

    # titles
    p4 = sub.add_parser("titles", help="Build job titles embeddings")
    p4.add_argument("csv", help="Path to job titles CSV (Job Title Id,Name)")

    # add
    p5 = sub.add_parser("add", help="Add entries to an existing collection")
    p5.add_argument(
        "--collection", "-c", required=True, help="Collection name (skills/colleges/companies/titles)"
    )
    p5.add_argument(
        "--json", "-j",
        required=True,
        help="Path to JSON file containing list of {id:..., text:...}"
    )

    args = parser.parse_args()

    try:
        if args.cmd == "skills":
            build_embeddings(args.csv, "s_no", "name", "skills", normalize_skill)
        elif args.cmd == "colleges":
            build_embeddings(args.csv, "CollegeID", "College Name", "colleges", normalize_generic)
        elif args.cmd == "companies":
            build_embeddings(args.csv, "Company ID", "Company", "companies", normalize_generic)
        elif args.cmd == "titles":
            build_embeddings(args.csv, "Job Title Id", "Name", "titles", normalize_generic)
        elif args.cmd == "add":
            with open(args.json, 'r', encoding='utf-8') as f:
                entries = json.load(f)
            add_to_collection(args.collection, entries)
        else:
            parser.error("Unknown command")
    except Exception as e:
        logging.error("Error: %s", e, exc_info=False)
        exit(1)


if __name__ == "__main__":
    main()
