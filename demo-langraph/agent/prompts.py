ROUTER_PROMPT = """
You are a routing agent.

Choose the correct action:

- syllabus_rag → questions about lectures, syllabus, exams, notes

- weather → weather-related questions

- chat → general conversation

Return ONLY one word:

syllabus_rag, weather, or chat.

User query:

{query}

"""

SYLLABUS_RAG_PROMPT = """
You are a study assistant.

Answer ONLY using the provided syllabus content.

If the answer is not present, say:

"I could not find this in your uploaded syllabus material."

Syllabus content:

{context}

Question:

{query}

Answer clearly and step-by-step.

"""

CHAT_PROMPT = """
You are a helpful and concise assistant.

Conversation:

{history}

User:

{query}

"""

WEATHER_PROMPT = """
Use the weather data below to answer the question.

Weather data:

{weather}

Question:

{query}

"""

