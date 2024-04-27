import streamlit as st
from dotenv import load_dotenv
import re
import snowflake.connector
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


# Snowflake function to fetch course data
def get_courses_from_snowflake():
    # Placeholder for Snowflake connection details
    user = 'SREYASRAI25'
    password = 'Shreyasrai1997@'
    account = 'mupkhwv-vcb24534'
    warehouse = 'WH_CC'
    database = 'CC'
    schema = 'CC_SCHEMA'

    # SQL query to fetch course data
    query = """
    SELECT COURSE_ID, COURSE_NAME, CREDIT_HOURS, COURSE_DESC, PREREQUISITE,
           PROFESSOR_NAME, RATING, POSITIVE_REVIEWS, NEGATIVE_REVIEWS
    FROM CC.CC_SCHEMA.CC_TABLE2
    """

    courses_data = []
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
        for row in rows:
            course_info = f"""Course ID: {row[0]}
            Course Name: {row[1]}
            Credit Hours: {row[2]}
            Course Description: {row[3]}
            Prerequisite: {row[4]}
            Professor Name: {row[5]}
            Rating: {row[6]}
            Positive Reviews: {row[7]}
            Negative Reviews: {row[8]}\n"""
            courses_data.append(course_info)
        all_courses_text = "\n".join(courses_data)
        return all_courses_text
    finally:
        cur.close()
        ctx.close()


def redact_pii(text):
    # Patterns for different types of PII
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\d{3}[-.]?\d{3}[-.]?\d{4}|\(\d{3}\)\s*\d{3}[-.]?\d{4})\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    }
    placeholder = '[REDACTED]'
    for _, pattern in patterns.items():
        text = re.sub(pattern, placeholder, text)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)
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


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Course Data", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history     = None

    st.header("Chat about Course Data :books:")
    user_question = st.text_input("Ask a question about the courses:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Fetch Course Data")
        if st.button("Fetch and Process Data"):
            with st.spinner("Fetching data from Snowflake..."):
                # Fetch course data from Snowflake
                raw_text = get_courses_from_snowflake()

                # Split the fetched data into chunks for processing
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversation chain with the fetched and processed data
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Data fetched and processed successfully!")

if __name__ == '__main__':
    main()