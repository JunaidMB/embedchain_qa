from embedchain import App
from embedchain.config import AppConfig, AddConfig, QueryConfig, ChunkerConfig
from string import Template
from dotenv import load_dotenv

load_dotenv()

# App Config
#config = AppConfig(log_level="INFO", collection_name = "fine_tune_blog_bot")
#fine_tune_bot = App(config = config)
fine_tune_bot = App()

# Add Config
chunker_config = ChunkerConfig(chunk_size=500, chunk_overlap=25, length_function=len)
fine_tune_bot.add("https://bdtechtalks.com/2023/07/10/llm-fine-tuning/", "web_page", config = AddConfig(chunker=chunker_config))

# Query Config
prompt_template = Template("""
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
$context
Question: $query
Helpful Answer:""")

query_config = QueryConfig(template=prompt_template, temperature=0)

response = fine_tune_bot.query("What is the difference between supervised and unsupervised finetuning", config=query_config)
print(response)

response = fine_tune_bot.query("What is Reinforcement learning from human feedback", config=query_config)
print(response)

response = fine_tune_bot.chat("Repeat all previous responses to questions asked so far")
print(response)
