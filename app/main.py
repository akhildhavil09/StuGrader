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
from .ml_model.grading_model import AIGrader

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

def process_document(content: bytes, filename: str) -> str:
    """Convert document content to text based on file type."""
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            return ' '.join(page.extract_text() for page in pdf_reader.pages)
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(content))
            return ' '.join(paragraph.text for paragraph in doc.paragraphs)
        else:
            return content.decode('utf-8')
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        raise Exception(f"Could not process {filename}. Make sure it's a valid PDF or DOCX file.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/analyze")
async def analyze_assignment(
    rubric: UploadFile = File(...),
    assignment: UploadFile = File(...)
):
    # Add file size check (5MB limit)
    MAX_FILE_SIZE = 5 * 1024 * 1024
    
    try:
        print(f"Received files: {rubric.filename}, {assignment.filename}")
        
        # Check and read rubric
        print("Reading rubric file...")
        rubric_content = await rubric.read()
        if len(rubric_content) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={"error": "Rubric file too large. Please keep files under 5MB."}
            )
            
        # Check and read assignment
        print("Reading assignment file...")
        assignment_content = await assignment.read()
        if len(assignment_content) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={"error": "Assignment file too large. Please keep files under 5MB."}
            )

        print("Converting files to text...")
        rubric_text = process_document(rubric_content, rubric.filename)
        assignment_text = process_document(assignment_content, assignment.filename)

        print("Initializing grader...")
        grader = AIGrader()
        
        print("Starting analysis...")
        results = grader.analyze_rubric_and_assignment(rubric_text, assignment_text)

        print("Analysis complete")
        return JSONResponse(content=results)

    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)