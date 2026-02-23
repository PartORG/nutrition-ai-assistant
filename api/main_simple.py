from fastapi import FastAPI

app = FastAPI(title="Nutrition AI Assistant")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/test")
async def test():
    return {"message": "API is working!"}