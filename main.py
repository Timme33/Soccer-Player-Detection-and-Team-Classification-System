from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from SoccerDetect import run_cv_pipeline

app = FastAPI()

# Allow your frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    input_path = f"temp/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    output1, output2 = run_cv_pipeline(input_path)
    return {
        "annotated1": f"/image/{os.path.basename(output1)}",
        "annotated2": f"/image/{os.path.basename(output2)}",
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    return FileResponse(filename)