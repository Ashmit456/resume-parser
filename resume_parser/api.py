import os
import json
import time
import string
import asyncio
import tempfile
import traceback
import random
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
from jsonschema import validate, ValidationError
from utils import convert_resume

ASSISTANT_ID_FILE = "assistant_3_id.txt"
SCHEMA_FILE = "schema.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with open(ASSISTANT_ID_FILE, "r") as f:
            app.state.assistant_id = f.read().strip()
        with open(SCHEMA_FILE, "r") as f:
            app.state.schema = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load resources: {str(e)}")
    yield

app = FastAPI(lifespan=lifespan)

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def clean_text(raw: str) -> str:
    printable = set(string.printable)
    filtered = "".join(ch for ch in raw if ch in printable)
    lines = [line.strip() for line in filtered.splitlines()]
    text = "\n".join(line for line in lines if line)
    return " ".join(text.split())

async def process_cv(file_path: str, assistant_id: str, schema: dict) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        raw_text = extract_text_from_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    cv_text = clean_text(raw_text)

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Please parse this CV:\n\n{cv_text}"
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    final_result = None
    try:
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id, 
                run_id=run.id
            )

            if run.status == "requires_action":
                actions = getattr(run, 'required_action', None)
                submits = getattr(actions, 'submit_tool_outputs', None) if actions else None
                tool_calls = getattr(submits, 'tool_calls', None) if submits else None

                if tool_calls:
                    for tool_call in tool_calls:
                        if tool_call.function.name == "fill_candidate_profile":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                validate(instance=args, schema=schema)
                                final_result = convert_resume(args)
                                
                                client.beta.threads.runs.submit_tool_outputs(
                                    thread_id=thread.id,
                                    run_id=run.id,
                                    tool_outputs=[{
                                        "tool_call_id": tool_call.id,
                                        "output": json.dumps(args)
                                    }]
                                )
                            except ValidationError as ve:
                                raise HTTPException(
                                    status_code=422,
                                    detail=f"Schema validation failed: {str(ve)}"
                                )
            elif run.status == "completed":
                break
            elif run.status in ("failed", "cancelled", "expired"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Assistant run failed with status: {run.status}"
                )
            else:
                await asyncio.sleep(1)

    except Exception:
        raise

    if final_result is None:
        raise HTTPException(
            status_code=500,
            detail="No valid profile data generated"
        )
    
    return final_result

@app.post("/extract")
async def parse_cv_endpoint(file: UploadFile = File(...)):
    tmp_path = None
    max_retries = 3
    base_delay = 1
    attempt = 0
    
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        last_exception = None
        while attempt < max_retries:
            try:
                result = await process_cv(
                    tmp_path,
                    app.state.assistant_id,
                    app.state.schema
                )
                return result
            except HTTPException as he:
                if 500 <= he.status_code < 600:
                    last_exception = he
                else:
                    raise
            except Exception as e:
                last_exception = e

            attempt += 1
            if attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.2)
                await asyncio.sleep(sleep_time)
        
        if isinstance(last_exception, HTTPException):
            raise last_exception
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Request failed after {max_retries} attempts"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True
    )
