import os
import sys
import shutil
import streamlit as st  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from modules.doc_processing import load_data, split_data, save_embeddings

base_dir = os.path.dirname(os.path.abspath(__file__))



# Streamlit interface
st.title("Chat with Document")

st.write("Upload PDF or DOCX files to build your dynamic knowledge base")

# File uploader for PDF and ZIP files
uploaded_file = st.file_uploader(
    "Upload a PDF, DOCX",
    type=["pdf", "docx"],  # Updated to accept 'zip' files
    accept_multiple_files=False,
)


# Specify the permanent upload directory
upload_dir = os.path.join(base_dir, "data")

index_path = os.path.join(base_dir, "data", "vector_store.faiss", "index.faiss")

index_embedding = os.path.join(base_dir, "data", "vector_store.faiss")


# Check and create the directory if not present
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    

    
uploaded_files = os.listdir(upload_dir)
    
if uploaded_files:
    st.write("Currently available files in the directory:")
    for file in uploaded_files:
        # Check if the file is PDF or DOCX
        if file.endswith(".pdf") or file.endswith(".docx"):
            st.write(f"- {file}")
else:
    st.write("No files currently uploaded.")
    
    
    

rag_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, max_output_tokens=8192)


def load_embeddings_data(index_embedding):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.load_local(index_embedding, embeddings, allow_dangerous_deserialization=True)
    
    return db



if uploaded_file:
    # Define the save path
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File uploaded successfully!")
    
    # Trigger background processing if embeddings don't already exist
    if not os.path.exists(index_path):
        with st.spinner("Processing your files..."):
            documents = load_data()
            if documents:
                text_chunks = split_data(documents)
                save_embeddings(text_chunks)
                st.success("You can now query the data.")
            else:
                st.warning("No valid documents found to process.")
else:
    st.write("Upload a file to create a knowledge base.")
    
    
if os.path.exists(index_path):
    # If vector database exists, proceed with querying
    rag_prompt = """ 
    You are an advanced assistant designed to help users query and interpret complex industry-standard documents, technical guidelines, and compliance instructions. Your role is to provide precise, contextually relevant, and actionable answers based solely on the document's content.

    Guidelines for Responding to Queries:

    1. **Understand User Intent**:
        - Accurately interpret user queries based on the provided context and question.
        - If the query is ambiguous, ask clarifying questions to refine the user's intent.
        - Determine if the query requires a short answer or a detailed response:
            - Short Answer: If the question is direct or seeks a concise explanation (e.g., 'What is section 3.1?'),
              provide a brief, specific answer.
            - Detailed Answer: If the query asks for an explanation, comparison, or breakdown 
              (e.g., 'Can you explain the steps for compliance in section 4?'), provide a comprehensive and structured response, possibly breaking it into manageable part
        - When the user provides detailed information about their product or scenario:
            - Identify the relevant aspects of the product (e.g., specifications, use case, compliance needs) to focus your response.
            - Match this information to the applicable guidelines or standards in the document.

    2. **Document-Specific Responses**:
        - Use the content of the provided documents to construct fact-based answers.
        - Reference section titles, numbers, or metadata for clarity (e.g., 'Section 5.3: Safety Requirements').
        - If the requested information isn’t explicitly found in the document, clearly state, 'This information is not available in the provided documents.'

    3. **Output Formatting and Style**:
        - Structure responses logically and professionally. Use bullet points, numbered steps, or tables when answering procedural or comparison-based questions.
        - Maintain industry-appropriate terminology and tone while ensuring accessibility for users with varying expertise levels.
        - For complex queries, break down the response into manageable parts, summarizing key points before diving into details.

    4. **Handle Contextual Complexity**:
        - Address multi-step or compound queries comprehensively.
        - Summarize related sections or cross-reference similar standards if it enhances user understanding.

    5. **Metadata and Enhanced Retrieval**:
        - Leverage document metadata, such as section titles, tags, or keywords, to refine and enhance the relevance of your response.
        - Indicate the source section or subsection for every provided answer to improve traceability.

    6. **Clarify Limitations**:
        - If a query requires interpretation or decisions beyond the document’s content, clearly state that your responses are fact-based and do not provide subjective opinions.
        - Avoid making assumptions; if a query requires external knowledge, state this explicitly.

    Response Format Guidelines:
        - **Bullet Points**: Use bullet points for clarity when listing multiple details or points.
        - **Direct Answers**: Provide direct answers for questions about specific clauses, sections, or standards.
        - **Keep It Concise**: Stick to brief and relevant responses, maintaining focus on the user’s question.
        - **For Detailed Responses**: Provide thorough explanations, breaking down steps or sections where needed.

    Context:  
    {context}

    Question:  
    {question}
    
    """
    
    
    prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "question"])

    db = load_embeddings_data(index_embedding)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=rag_model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    # Query input box
    question = st.text_input("Ask a question related to the documents:")

    if question:
        with st.spinner("Fetching response..."):
            response = qa_chain.invoke({"query": question})
            st.write("Response:")
            st.write(response['result'])
            
else:
    # If vector database does not exist, prompt the user to upload content
    st.warning("No Knowledgebase found. Please upload document.")
    
    
    
    
if st.button("Clear All Uploaded Data"):
    
    embedding_path = os.path.join(base_dir, "..","data", "vector_store.faiss")
    
    if os.path.exists(upload_dir):
        for file_name in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file_name)
            # Check if it is a file and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    if os.path.exists(embedding_path):
        shutil.rmtree(embedding_path)
    
    st.success("All uploaded files have been cleared!")
