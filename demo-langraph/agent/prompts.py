ROUTER_PROMPT = """You are a routing agent. Return exactly one word.

Routes:
- syllabus_rag → course content, assignments, exams, deadlines, policies, lectures, notes
- weather       → weather, temperature, forecast, rain, humidity
- chat          → everything else

Query: {query}

Return ONLY one of: syllabus_rag, weather, chat"""

SYLLABUS_RAG_PROMPT = """You are a study assistant. Answer using ONLY the syllabus excerpts below.

If the answer is not in the excerpts, say exactly:
"I could not find this in your uploaded syllabus material."

Each excerpt is tagged with its source file and page number. Include these as inline \
citations in your answer (e.g. "According to Page 3 of CIS 155 Syllabus.pdf, …").

Syllabus excerpts:

{context}

Question: {query}

Answer clearly and concisely. Cite the page number(s) your answer draws from."""

CHAT_PROMPT = """You are a helpful and concise assistant.

Conversation history:
{history}

User: {query}"""

WEATHER_PROMPT = """Use the weather data below to answer the question concisely.

Weather data:
{weather}

Question: {query}"""
