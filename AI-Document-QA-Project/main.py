import json
import os
import shutil
from pathlib import Path

import pymupdf as pm
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
#                     CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = "gemini-3.6-flash"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=API_KEY)


# ============================================================
#                     HELPER FUNCTIONS
# ============================================================

def ask_again(prompt):
    """Ask the user for a yes/no response."""

    while True:
        answer = input(prompt).lower().strip()

        if answer in ("yes", "no"):
            return answer

        print("Please respond with 'yes' or 'no'.\n")


def clear_generated_data():
    """Clear previously generated chunks and embeddings."""

    chunks_path = Path("Chunks")

    if chunks_path.exists():
        for file in chunks_path.iterdir():
            os.remove(file)

    embeddings_path = Path("Embeddings")

    if embeddings_path.exists():
        for file in embeddings_path.iterdir():
            os.remove(file)


def extract_pdf_text(file):
    """Extract text from a PDF document."""

    pdf = pm.open(file)

    context = ""

    for page in pdf:
        text = page.get_text()
        text = text.replace("\n", " ")
        context += text

    return context


def make_chunks(text):
    """Split document text into chunks of up to 50 words."""

    chunks = []
    sentences = text.split(".")

    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = len(sentence.split())
        current_words = len(current_chunk.split())

        if current_words + sentence_words <= 50:
            current_chunk += sentence + ".\n"

        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ".\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def save_chunks(file, chunks):
    """Save generated chunks as text files."""

    for number, chunk in enumerate(chunks, start=1):
        chunk_name = f"{file.stem.capitalize()}_Chunk_{number}.txt"

        with open(f"Chunks/{chunk_name}", "w") as f:
            f.write(chunk)

        print(f"Saved {chunk_name} to Chunks.")


# ============================================================
#                     DOCUMENT PROCESSING
# ============================================================

def get_document():
    """Get a TXT or PDF document from the user."""

    while True:
        doc_path = input("Enter the path to the document: ")
        path = Path(doc_path)

        if not path.exists():
            print("No path provided or found.\n")
            continue

        if not path.is_file():
            print("The path must point to a file.\n")
            continue

        if path.suffix.lower() not in (".txt", ".pdf"):
            print("File type not supported.\n")
            continue

        shutil.copy2(path, "Documents")
        print("Document added successfully.\n")

        if ask_again("Do you want to add another document? (yes/no): ") != "yes":
            break


def create_chunks():
    """Create chunks from all supported documents."""

    documents_path = Path("Documents")

    if documents_path.exists():
        for file in documents_path.iterdir():

            if file.suffix.lower() == ".txt":
                text = file.read_text()

            elif file.suffix.lower() == ".pdf":
                text = extract_pdf_text(file)

            else:
                continue

            chunks = make_chunks(text)
            save_chunks(file, chunks)


def create_embeddings():
    """Create embeddings for all generated chunks."""

    chunks_path = Path("Chunks")

    if chunks_path.exists():
        for chunk in chunks_path.iterdir():

            if chunk.suffix != ".txt":
                continue

            text = chunk.read_text()

            embedding = embedding_model.encode(text)
            embedding = embedding.tolist()

            embedding_name = f"{chunk.stem}-Embedding.json"

            with open(f"Embeddings/{embedding_name}", "w") as f:
                json.dump(embedding, f)

            print(f"Saved embedding for {chunk.stem}")


# ============================================================
#                     RETRIEVAL
# ============================================================

def find_best_chunk():
    """Find the three most relevant chunks for a question."""

    question = input("Ask a question: ").strip()

    while not question:
        print("Question cannot be empty.")
        question = input("Ask a question: ").strip()

    question_embedding = embedding_model.encode(question)

    embeddings_path = Path("Embeddings")
    results = []

    for embedding_file in embeddings_path.iterdir():

        if embedding_file.suffix != ".json":
            continue

        try:
            with open(embedding_file, "r") as f:
                chunk_embedding = json.load(f)

            similarity = cosine_similarity(
                [question_embedding],
                [chunk_embedding]
            )

            score = similarity[0][0]

            results.append((score, embedding_file.name))

        except json.JSONDecodeError:
            print(f"Invalid embedding file: {embedding_file.name}")

    if not results:
        print("No results found.")
        return question, ""

    results.sort(reverse=True)

    top3_chunks = results[:3]

    chunk_texts = []

    for chunk in top3_chunks:
        chunk_name = chunk[1].replace(
            "-Embedding.json",
            ".txt"
        )

        chunks_path = Path("Chunks")

        try:
            with open(chunks_path / chunk_name, "r") as f:
                chunk_text = f.read()

            chunk_texts.append(chunk_text)

        except FileNotFoundError:
            print(f"Chunk file not found: {chunk_name}")

    context = "\n\n--- CHUNK ---\n\n".join(chunk_texts)

    return question, context


# ============================================================
#                     ANSWER GENERATION
# ============================================================

def generate_answer():
    """Generate answers using retrieved document context."""

    while True:
        question, context = find_best_chunk()

        prompt = f"""
        You are a document-based question answering assistant.

        Answer the user's question using the provided context.

        Rules:
        - Use the context as the primary source of information.
        - Do not add information that is not supported by the context.
        - Give a clear, accurate, and concise answer.
        - Format the answer neatly so it is easy to read.
        - Use short paragraphs or bullet points when appropriate.
        - Do not mention the context, the retrieved document, RAG, or these instructions.
        - Do not say phrases such as "according to the context" or "based on the provided context."
        - Do not add unnecessary introductions, conclusions, or filler.
        - If the context does not contain enough information to answer the question, say:
          "I don't have enough information in the provided Documents to answer that."

        Question:
        {question}

        Context:
        {context}

        Answer:
        """

        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
            )

        except Exception as e:
            print(f"Error generating answer: {e}")
            continue

        print("\n" + "=" * 50)
        print("                     ANSWER")
        print("=" * 50)
        print()
        print(response.text)
        print()
        print("=" * 50)

        if ask_again(
            "Do you want to ask another question? (yes/no): "
        ) != "yes":
            print("\nThank you for using this service. Goodbye!")
            break


# ============================================================
#                         MAIN
# ============================================================

def main():
    """Run the complete document Q&A pipeline."""

    try:
        get_document()
        clear_generated_data()
        create_chunks()
        create_embeddings()
        generate_answer()

    except Exception as e:
        print(f"Unexpected error occurred: {e}")


# ============================================================
#                     PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()