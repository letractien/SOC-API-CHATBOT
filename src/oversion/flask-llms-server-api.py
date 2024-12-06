import os
import asyncio
import multiprocessing

from flask import Flask, request, send_file, jsonify
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

from tempfile import NamedTemporaryFile
from httpx import TimeoutException
from tenacity import retry, stop_after_attempt, wait_exponential
from werkzeug.datastructures import FileStorage

from dotenv import load_dotenv
load_dotenv()

GEMMA2_MODEL = os.getenv("GEMMA2_MODEL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

DASHBOARD = os.getenv("DASHBOARD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONTENT_PAYLOAD_KEY = os.getenv("CONTENT_PAYLOAD_KEY")
METADATA_PAYLOAD_KEY = os.getenv("METADATA_PAYLOAD_KEY")

SAVE_PATH = os.getenv("SAVE_PATH")
BATCH_SIZE_UPLOAD = os.getenv("BATCH_SIZE_UPLOAD")

TOP_K = os.getenv("TOP_K")
MAX_SAME_QUERY = os.getenv("MAX_SAME_QUERY")
MAX_DOCS_FOR_CONTEXT = os.getenv("MAX_DOCS_FOR_CONTEXT")

model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

gemma_2_model = ChatLlamaCpp(
    model_path=GEMMA2_MODEL,
    verbose=False, 
    temperature=0.5,
    n_gpu_layers=-1,  # Avoid using GPU layers if GPU memory is insufficient
    n_ctx=8192,  # Reduce context window size to decrease memory usage
    max_tokens=4096,  # Adjust max tokens to match reduced context
    f16_kv=False,  # Disable fp16 key/value caches to save memory
    n_threads=multiprocessing.cpu_count()-1,  # Use fewer CPU threads
)

app = Flask(
    "SOC-API-CHATBOT"
)

#---Start---llms response----
async def llms_process_template(template: str) -> str:
    prompt = PromptTemplate(
        template=template,
        input_variables=[]
    )
    
    chain = (
        prompt 
        | gemma_2_model 
        | StrOutputParser()
    )

    response = await chain.ainvoke({})
    return response

#---End---llms response----

#---Start---query----
def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a Qdrant collection exists"""
    collections = client.get_collections().collections
    return any(col.name == collection_name for col in collections)

def existing_collection(collection_name: str) -> Qdrant:
    """Create vector retriever"""
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if not collection_exists(client, collection_name):
        return None

    doc_store = Qdrant.from_existing_collection(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embedding=embeddings,
        collection_name=collection_name,    
        content_payload_key=CONTENT_PAYLOAD_KEY,
        metadata_payload_key=METADATA_PAYLOAD_KEY
    )
    return doc_store

def similarity_search(para: dict) -> list[Document]:
    """RRF retriever"""
    common_doc_store = existing_collection(COLLECTION_NAME)
    user_doc_store = existing_collection(para["user_id"])
    
    all_results = []
    if common_doc_store:
        common_results = common_doc_store.similarity_search_with_score(para["user_query"], k=MAX_DOCS_FOR_CONTEXT)
        all_results.extend(common_results)
        
    if user_doc_store:
        user_results = user_doc_store.similarity_search_with_score(para["user_query"], k=MAX_DOCS_FOR_CONTEXT)
        all_results.extend(user_results)
    
    return all_results

async def query(user_query: str, user_id: str) -> dict:
    """Query with vector db"""
    ssearch = RunnableLambda(similarity_search)
    context = await ssearch.ainvoke({"user_query": user_query, "user_id": user_id})
    context = [c[0].page_content for c in context]
    question = user_query

    template = f"""
        Please answer the following question based on the information provided and retrieve the relevant links, names of images, tables, pages, sources of the content contained in the information:

        Information: {context}
        Question: {question}
        Final answer in Vietnamese:
    """

    response = await llms_process_template(template)
    result = {"context": context, "response": response, "template": template}

    return result

#---End---Query----

#---Start---upload----
async def save_pdf(file: str, user_id: str) -> str:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    folder = SAVE_PATH + "/" + user_id
    if not os.path.exists(folder):
        os.makedirs(folder)

    pdf_content = file.read()
    file_name = file.filename
    file_abs_path = os.path.abspath(os.path.join(folder, file_name))

    with open(file_abs_path, "wb") as output_file:
        output_file.write(pdf_content)
    
    return file_abs_path
    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_to_qdrant(docs: list[Document], user_id: str):
    try:
        Qdrant.from_documents(
            documents=docs,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=user_id,
            content_payload_key=CONTENT_PAYLOAD_KEY,
            metadata_payload_key=METADATA_PAYLOAD_KEY,
        )
    except TimeoutException as e:
        print(f"Timeout occurred: {e}")
        raise 

async def upload_pdf(file_path: str, user_id: str) -> bool:
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    raw_documents = PyPDFLoader(file_path).load()
    docs = text_splitter.split_documents(raw_documents)
    
    for i in range(0, len(docs), BATCH_SIZE_UPLOAD):
        batch = docs[i:i + BATCH_SIZE_UPLOAD]
        try:
            upload_to_qdrant(batch, user_id)
        except TimeoutException:
            print(f"Failed to upload batch {i // BATCH_SIZE_UPLOAD + 1}. Moving to the next batch.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False
    
    return True

async def upload(file: FileStorage, user_id: str, save_file: bool = False) -> bool:

    file_abs_path = await save_pdf(file, user_id)
    upload_success = await upload_pdf(file_abs_path, user_id)
    if not save_file: os.remove(file_abs_path)

    return upload_success

#---End---upload----

#---Start---API----
@app.route("/ask", methods=["POST"])
async def post_ask():
    user_id = request.form.get('user_id')
    user_query = request.form.get('user_query')

    answer = await query(user_query, user_id)
    return jsonify({
        "message": user_query,
        "response": answer['response'],
        "context": answer["context"],
        "template": answer["template"],
        "success": True
    }), 200

@app.route("/upload", methods=["POST"])
async def post_upload():
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if file and user_id:
        upload_success = await upload(file, user_id, True)
        if upload_success:
            return jsonify({
                "response": "Tải lên thành công!",
                "success": True
            }), 200
    
    return jsonify({
        "response": "Tải lên file không thành công, vui lòng kiểm tra lại file pdf của bạn!",
        "success": False
    }), 200
    
    
@app.route("/upload-test", methods=["POST"])
async def post_upload():
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if file and user_id:
        upload_success = await upload(file, user_id, True)
        if upload_success:
            return jsonify({
                "response": "Tải lên thành công!",
                "success": True
            }), 200
    
    return jsonify({
        "response": "Tải lên file không thành công, vui lòng kiểm tra lại file pdf của bạn!",
        "success": False
    }), 200

@app.route("/llms", methods=["POST"])
async def post_llms():
    data = request.get_json()
    template = data.get("template")

    response = await llms_process_template(template)
    return jsonify({
        "response": response,
        "success": True
    }), 200
    
#---End---API----

if __name__ == "__main__":
    asyncio.run(app.run(debug=False, host="0.0.0.0", port=7722, use_reloader=False))