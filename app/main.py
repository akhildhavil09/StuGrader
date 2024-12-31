from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import io
from docx import Document
import PyPDF2
import traceback
import torch
from transformers import AutoTokenizer, AutoModel
from ml_model.grading_model import AIGrader  # Import our custom grader

def process_document(content: bytes, filename: str) -> str:
    """Convert document content to text based on file type."""
    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        return ' '.join(page.extract_text() for page in pdf_reader.pages)
    elif filename.endswith('.docx'):
        doc = Document(io.BytesIO(content))
        return ' '.join(paragraph.text for paragraph in doc.paragraphs)
    else:
        return content.decode('utf-8')

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Mount static directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# app/main.py - Update the analyze endpoint
@app.post("/analyze")
async def analyze_assignment(
    rubric: UploadFile = File(...),
    assignment: UploadFile = File(...)
):
    try:
        print("Starting assignment analysis process...")
        
        rubric_text = await rubric.read()
        assignment_text = await assignment.read()

        # Convert bytes to text
        rubric_text = process_document(rubric_text, rubric.filename)
        assignment_text = process_document(assignment_text, assignment.filename)

        # Initialize and use grader
        grader = AIGrader()
        results = grader.analyze_rubric_and_assignment(rubric_text, assignment_text)

        return results

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)