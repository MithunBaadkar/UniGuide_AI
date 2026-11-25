from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load University Policy / Handbook PDF
loader = PyPDFLoader("university_handbook.pdf")
documents = loader.load()

# 2. Chunk the text into manageable pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Generate Embeddings & Store in Vector Database (FAISS)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Load Open Model Directly in Python (No API Key, No Ollama Server)
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    device_map="auto",
    pipeline_kwargs={"max_new_tokens": 256, "temperature": 0.1, "return_full_text": False},
)

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {input}\nAnswer:"
)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(
    vectorstore.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain,
)

# 5. Query the System
query = "What is the minimum attendance requirement to sit for semester exams?"
response = qa_chain.invoke({"input": query})

print("Answer:", response["answer"].strip())
print("Source Page:", response["context"][0].metadata)