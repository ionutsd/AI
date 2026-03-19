

# rag_chain.py
import re
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load full dataframe for exact aggregations
df = pd.read_excel("Superstore.xlsx")

# Embeddings — must match ingest!
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="",
    temperature=0.1
)

# Prompt
prompt = PromptTemplate.from_template("""You are a senior business data analyst for Tableau Superstore data.
You have been given a sample of relevant transaction records from the database.
Use these records to provide analytical insights and reasoning.

Even if the sample is small, you should:
- Identify patterns in the data provided
- Make reasonable business observations
- Suggest what the data might indicate
- Be clear when you are inferring vs stating facts

Context (sample Superstore records):
{context}

Question: {question}

Provide a thoughtful analytical response based on the data above:""")

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Filter extraction ─────────────────────────────────────────────────────────

def extract_filters_from_context(question: str) -> dict:
    filters = {}
    match = re.search(r'\[Dashboard Context: Active filters: (.+?)\]', question)
    if not match:
        return filters
    filter_text = match.group(1)
    for part in filter_text.split(' | '):
        if ':' in part:
            field, values = part.split(':', 1)
            field = field.strip()
            values = [v.strip() for v in values.split(',')]
            filters[field] = values
    return filters


def apply_filters_to_df(base_df, filters: dict):
    filtered = base_df.copy()
    col_map = {
        'Region': 'Region',
        'Category': 'Category',
        'Sub-Category': 'Sub-Category',
        'Segment': 'Segment',
        'State': 'State',
        'City': 'City',
        'Ship Mode': 'Ship Mode',
    }
    for field, values in filters.items():
        col = col_map.get(field)
        if col and col in filtered.columns:
            filtered = filtered[filtered[col].isin(values)]
            print(f"Applied filter: {col} in {values} → {len(filtered)} rows remaining")
    return filtered


# ── Pandas aggregation layer ──────────────────────────────────────────────────

def calculate_stats(question: str, filtered_df=None) -> str:
    data = filtered_df if filtered_df is not None else df
    q = question.lower()

    try:
        if "profit" in q and "category" in q:
            result = data.groupby("Category")["Profit"].sum().sort_values(ascending=False)
            return f"Total profit by category:\n{result.to_string()}"

        if "profit" in q and "region" in q:
            result = data.groupby("Region")["Profit"].sum().sort_values(ascending=False)
            return f"Total profit by region:\n{result.to_string()}"

        if "profit" in q and "segment" in q:
            result = data.groupby("Segment")["Profit"].sum().sort_values(ascending=False)
            return f"Total profit by segment:\n{result.to_string()}"

        if "profit" in q and "state" in q:
            result = data.groupby("State")["Profit"].sum().sort_values(ascending=False)
            return f"Total profit by state:\n{result.to_string()}"

        if "profit" in q and ("sub-category" in q or "sub category" in q):
            result = data.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
            return f"Total profit by sub-category:\n{result.to_string()}"

        if "profit" in q and "ship" in q:
            result = data.groupby("Ship Mode")["Profit"].sum().sort_values(ascending=False)
            return f"Total profit by ship mode:\n{result.to_string()}"

        if "sales" in q and "category" in q:
            result = data.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            return f"Total sales by category:\n{result.to_string()}"

        if "sales" in q and "region" in q:
            result = data.groupby("Region")["Sales"].sum().sort_values(ascending=False)
            return f"Total sales by region:\n{result.to_string()}"

        if "sales" in q and "segment" in q:
            result = data.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
            return f"Total sales by segment:\n{result.to_string()}"

        if "sales" in q and "state" in q:
            result = data.groupby("State")["Sales"].sum().sort_values(ascending=False)
            return f"Total sales by state:\n{result.to_string()}"

        if "sales" in q and ("sub-category" in q or "sub category" in q):
            result = data.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
            return f"Total sales by sub-category:\n{result.to_string()}"

        if "sales" in q and "ship" in q:
            result = data.groupby("Ship Mode")["Sales"].sum().sort_values(ascending=False)
            return f"Total sales by ship mode:\n{result.to_string()}"

        if "quantity" in q and "category" in q:
            result = data.groupby("Category")["Quantity"].sum().sort_values(ascending=False)
            return f"Total quantity by category:\n{result.to_string()}"

        if "quantity" in q and "region" in q:
            result = data.groupby("Region")["Quantity"].sum().sort_values(ascending=False)
            return f"Total quantity by region:\n{result.to_string()}"

        if "discount" in q and "category" in q:
            result = data.groupby("Category")["Discount"].mean().sort_values(ascending=False)
            return f"Average discount by category:\n{result.to_string()}"

        if "discount" in q and "region" in q:
            result = data.groupby("Region")["Discount"].mean().sort_values(ascending=False)
            return f"Average discount by region:\n{result.to_string()}"

        if ("order" in q or "count" in q or "how many" in q) and "region" in q:
            result = data.groupby("Region")["Order ID"].nunique().sort_values(ascending=False)
            return f"Number of orders by region:\n{result.to_string()}"

        if ("order" in q or "count" in q or "how many" in q) and "category" in q:
            result = data.groupby("Category")["Order ID"].nunique().sort_values(ascending=False)
            return f"Number of orders by category:\n{result.to_string()}"

        if "how many" in q and ("order" in q or "staple" in q or "product" in q):
            product_match = re.search(r'for (.+?)(?:\?|$)', q)
            if product_match:
                product_name = product_match.group(1).strip()
                product_df = data[data['Product Name'].str.lower().str.contains(product_name, na=False)]
                count = product_df['Order ID'].nunique()
                rows = len(product_df)
                return f"Orders containing '{product_name}': {count} unique orders, {rows} line items"

        if "product" in q and ("top" in q or "best" in q or "highest" in q):
            result = data.groupby("Product Name")["Profit"].sum().sort_values(ascending=False).head(10)
            return f"Top 10 products by profit:\n{result.to_string()}"

        if "product" in q and ("worst" in q or "lowest" in q or "bottom" in q):
            result = data.groupby("Product Name")["Profit"].sum().sort_values(ascending=True).head(10)
            return f"Bottom 10 products by profit:\n{result.to_string()}"

        if "summary" in q or "overall" in q or "total" in q:
            total_sales = data["Sales"].sum()
            total_profit = data["Profit"].sum()
            total_orders = data["Order ID"].nunique()
            total_customers = data["Customer Name"].nunique()
            profit_margin = (total_profit / total_sales) * 100
            return (
                f"Summary:\n"
                f"Total Sales: ${total_sales:,.2f}\n"
                f"Total Profit: ${total_profit:,.2f}\n"
                f"Profit Margin: {profit_margin:.1f}%\n"
                f"Total Orders: {total_orders:,}\n"
                f"Total Customers: {total_customers:,}"
            )

    except Exception as e:
        print(f"Stats error: {e}")

    return None


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(question: str) -> dict:
    print(f"RAW QUESTION RECEIVED: {question}")  # Add this line
    # Extract dashboard filters from context prefix
    filters = extract_filters_from_context(question)
    print(f"FILTERS EXTRACTED: {filters}")  # Add this line
    # Apply filters to dataframe if any exist
    filtered_df = apply_filters_to_df(df, filters) if filters else df

    # Strip context prefix from question for cleaner processing
    clean_question = re.sub(r'\[Dashboard Context:.*?\]\n\n', '', question, flags=re.DOTALL).strip()

    # Try pandas aggregation with filtered data
    stats = calculate_stats(clean_question, filtered_df)
    if stats:
        filter_note = ""
        if filters:
            filter_note = f"(Filtered: {', '.join([f'{k}: {v}' for k, v in filters.items()])})\n\n"
        return {
            "answer": filter_note + stats,
            "sources_count": 0
        }

    # Fall back to RAG for qualitative questions
    answer = chain.invoke(question)
    return {
        "answer": answer,
        "sources_count": 8
    }
