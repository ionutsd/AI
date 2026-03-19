
# ingest.py — Run ONCE to build your ChromaDB knowledge base
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load Superstore CSV
df = pd.read_excel("Superstore.xlsx")
print(f"Loaded {len(df)} rows")

# Convert each row into a descriptive sentence
def row_to_text(row):
    return (
        f"Order {row['Order ID']}: {row['Category']} › {row['Sub-Category']} — "
        f"{row['Product Name']}. Region: {row['Region']}, State: {row['State']}, "
        f"City: {row['City']}, Customer: {row['Customer Name']}, "
        f"Segment: {row['Segment']}. "
        f"Sales: ${row['Sales']:.2f}, Profit: ${row['Profit']:.2f}, "
        f"Quantity: {row['Quantity']}, Discount: {row['Discount']:.0%}. "
        f"Ship Mode: {row['Ship Mode']}, Order Date: {row['Order Date']}."
    )

texts = df.apply(row_to_text, axis=1).tolist()
print(f"Sample row: {texts[0]}")

# Embedding model — runs locally, small footprint
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build and save ChromaDB
print("Building ChromaDB... (2-5 mins)")
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Done! {len(texts)} rows stored in ChromaDB.")
print("Never run this again unless your data changes!")
