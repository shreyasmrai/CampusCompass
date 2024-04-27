import streamlit as st
from dotenv import load_dotenv
import re
import snowflake.connector
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from htmlTemplates import css, bot_template, user_template
import json
import decimal

def get_courses_from_snowflake():
    # Snowflake connection details
    user = 'SREYASRAI25'
    password = 'Shreyasrai1997@'
    account = 'mupkhwv-vcb24534'
    warehouse = 'WH_CC'
    database = 'CAMPUSCOMPASS'
    schema = 'DEV'

    # SQL query to select courses
    query = """
    SELECT COURSE_ID, COURSE_NAME, CREDIT_HOURS, COURSE_DESC, PREREQUISITE,
           PROFESSOR_NAME, RATING, POSITIVE_REVIEW, NEGATIVE_REVIEW
    FROM CAMPUSCOMPASS.DEV.AWS_DATA
    """

    try:
        # Connect to Snowflake
        ctx = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        cur = ctx.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
            courses = []
            for row in rows:
                course_info = {
                    "course_id": str(row[0]),
                    "course_name": row[1],
                    "credit_hours": float(row[2]) if isinstance(row[2], decimal.Decimal) else row[2],
                    "course_description": row[3],
                    "prerequisite": row[4],
                    "professor_name": row[5],
                    "rating": float(row[6]) if isinstance(row[6], decimal.Decimal) else row[6],
                    "reviews": {
                        "positive": [review.strip() for review in (row[7] or "").split("\n") if review.strip()],
                        "negative": [review.strip() for review in (row[8] or "").split("\n") if review.strip()]
                    }
                }
                courses.append(course_info)
            courses_json = json.dumps(courses, indent=4)
            return courses_json
        finally:
            cur.close()
    except Exception as e:
        print(f"Failed to fetch data from Snowflake: {e}")
        return None
    finally:
        if 'ctx' in locals():
            ctx.close()

def process_course_data(courses):
    processed_courses = []
    for course in courses:
        course_details = (
            f"Course ID: {course['course_id']}\n"
            f"Course Name: {course['course_name']}\n"
            f"Credit Hours: {course['credit_hours']}\n"
            f"Course Description: {course['course_description']}\n"
            f"Prerequisite: {course['prerequisite']}\n"
            f"Professor Name: {course['professor_name']}\n"
            f"Rating: {course['rating']}\n"
            f"Positive Reviews: {'; '.join(course['reviews']['positive'])}\n"
            f"Negative Reviews: {'; '.join(course['reviews']['negative'])}"
        )
        processed_courses.append(course_details)
    return processed_courses

def redact_pii(text):
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\d{3}[-.]?\d{3}[-.]?\d{4}|\(\d{3}\)\s*\d{3}[-.]?\d{4})\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    }
    placeholder = '[REDACTED]'
    for _, pattern in patterns.items():
        text = re.sub(pattern, placeholder, text)
    return text

def get_vectorstore(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

def setup_prompt_selector():
    prompt_template = """As an expert University course guide, your role is to distill relevant information from the provided course details to specifically address the student's inquiry. Focus on extracting pertinent facts, insights, and implications that directly respond to the question posed. Avoid general summaries of the course information or including extraneous details not related to the query.

    Relevant Course Details:
    {context}

    Student's Question: {question}
    Targeted Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    system_template = """Your task is to provide a targeted and specific response to the student's question based on the course details provided. Concentrate on pulling out the most relevant information that directly answers the question. Sidestep the inclusion of broad summaries or unrelated facts.

    Relevant Course Details:
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("Please provide a targeted answer to this specific question: {question}"),
    ]
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

    PROMPT_SELECTOR = ConditionalPromptSelector(
        default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
    )
    return PROMPT_SELECTOR

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    prompt_selector = setup_prompt_selector()

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_selector.get_prompt(llm)}
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            user_message = user_template.replace("{{MSG}}", message.content)
            st.write(user_message, unsafe_allow_html=True)
        else:
            sanitized_content = redact_pii(message.content)
            bot_message = bot_template.replace("{{MSG}}", sanitized_content)
            st.write(bot_message, unsafe_allow_html=True)
            if 'additional_responses' in message:
                for additional_response in message['additional_responses']:
                    sanitized_additional = redact_pii(additional_response.content)
                    additional_message = bot_template.replace("{{MSG}}", sanitized_additional)
                    st.write(additional_message, unsafe_allow_html=True)

def create_course_chunks(courses):
    course_chunks = []
    for course in courses:
        course_chunk = json.dumps(course, ensure_ascii=False)
        course_chunks.append(course_chunk)
    return course_chunks

def get_vector_count(vectorstore):
    return vectorstore.index.ntotal

def create_and_print_course_chunks(courses):
    course_chunks = []
    try:
        for course in courses:
            course_chunk = json.dumps(course, ensure_ascii=False)
            course_chunks.append(course_chunk)
            print(f"Chunk length (in characters): {len(course_chunk)}")
    except Exception as e:
        print(f"Error during chunk creation: {e}")
    return course_chunks

def load_llm():
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama")
    return llm

def main():
    load_dotenv()
    st.set_page_config(page_title="CampusCompass", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("CampusCompass :books:")
    user_question = st.text_input("Ask a question about the courses:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Fetch Course Data")
        if st.button("Fetch and Process Data"):
            with st.spinner("Fetching data from Snowflake..."):
                courses_json = get_courses_from_snowflake()
                if courses_json:
                    courses = json.loads(courses_json)
                    processed_courses = process_course_data(courses)
                    text_chunks = create_and_print_course_chunks(processed_courses)
                    vectorstore = get_vectorstore(text_chunks)
                    vector_count = get_vector_count(vectorstore)
                    #st.write(f"Number of course chunks loaded in vector database: {vector_count}")
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Data fetched and processed successfully!")
                    else:
                        st.error("Failed to initialize vectorstore.")
                else:
                    st.error("Failed to fetch data.")

        job_roles = [
            "Chemical Process Engineer", "Energy Systems Engineer", "Safety Engineer", "Research Scientist",
            "Academic Researcher", "PhD Researcher", "Systems Engineer", "Circuit Design Engineer",
            "Power Systems Engineer", "Computer Engineer", "Safety Analyst", "Data Scientist",
            "Machine Learning Engineer", "Network Engineer", "Structural Engineer", "Civil Engineer",
            "Concrete and Steel Design Engineer", "Software Developer", "Systems Developer",
            "Project Manager", "Research and Development Engineer", "Dissertation Supervisor",
            "Environmental Engineer", "Climate Change Analyst", "Water and Air Quality Manager"
        ]
        selected_job_role = st.sidebar.selectbox("Select a Job Role:", job_roles)

        if st.sidebar.button("Find Courses"):
            if "conversation" in st.session_state:
                suggest_courses(selected_job_role)
            else:
                st.sidebar.error("Please load data first.")

def suggest_courses(job_role):
    context = f"Context information about job role: {job_role}"
    question = f"What courses are best suited for a {job_role}?"

    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 1:
            sanitized_content = redact_pii(message.content)
            st.sidebar.write(sanitized_content)

if __name__ == '__main__':
    main()
