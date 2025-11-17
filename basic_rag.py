from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I despise mangos.',
    'I hate Windows.',
    'Linux is a great operating system.'
]

vector_store = FAISS.from_texts(texts, embeddings)

# print(vector_store.similarity_search('Apples are my favorite food.', k=7))
# print(vector_store.similarity_search('Linux is a great operating system.', k=7))

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

retriever_tool = create_retriever_tool(retriever, name='kb_search', description='search small product / fruit database for information.')

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

agent = create_agent(
    model = llm,
    tools = [retriever_tool],
    system_prompt=(
        "You are a helpful assistant. For questions about Macs, apples, or laptops, "
        "First call the kb_search tool to retrieve context, then answer succinctly. Maybe you have to use it multiple times before answering."
),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})


print(result["messages"][-1].content)