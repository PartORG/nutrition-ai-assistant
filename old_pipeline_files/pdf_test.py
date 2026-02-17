from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_folder = Path("c:/Users/peter/Desktop/ds_ai/repo_folder/nutrition-ai-assistant/data_test/raw")
for pdf_file in pdf_folder.rglob("*.pdf"):
    try:
        print(f"\n📄 Testing: {pdf_file.name}...", end=" ")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        print(f"✅ OK ({len(docs)} pages)")
    except Exception as e:
        print(f"❌ FAILED: {e}")