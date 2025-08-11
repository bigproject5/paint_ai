from fastapi import FastAPI
from app.config import config

app = FastAPI(title="Paint AI API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Paint AI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
