import os
import json
from openai import OpenAI

SCHEMA_FILE = "schema.json"
ASSISTANT_NAME = "CV Parser v2"
MODEL = "gpt-4o"

INSTRUCTIONS = (
    "You are an expert AI assistant. Your task is to parse the provided CV text and convert it into a JSON object that strictly conforms to the provided JSON Schema. "
    "Do not invent any information. If a field's value cannot be found in the text, use an empty string \"\", an empty list [], or null as defined by the schema.\n\n"
    "Any city mentioned (e.g. current location, preferred location) must be represented as a Google-style location object: "
    "{\"name\": \"City Name\", \"placeid\": \"ChIxxxxxx\"}. GPT may supply the name, shortname, placeid and coordinates without a separate geocoding tool all these fields must be strictly filled.\n\n"
    "Differentiate between 'education' entries that belong in 'college' versus 'schooling': "
    "If the entry mentions degrees such as 'Bachelor', 'Master', 'PhD', 'MBA', 'B.Tech', 'Diploma', assign it to 'education'. "
    "If it mentions 'High School', 'Secondary School', 'HS', or 'School', assign it to 'schooling'.\n\n"
    "For skills and tools, follow these strict rules:\n"
    "- Skills include programming languages (e.g., Python, C++, JavaScript, Git), frameworks (e.g., TensorFlow, PyTorch, React), libraries, APIs, and technical knowledge areas.\n"
    "- Tools include development environments, editors, platforms, or utilities used to perform technical work (e.g., VS Code, Android Studio, JIRA, Docker, Kubernetes).\n"
    "- If unsure, do not guess — leave the entry out or set as an empty field as defined by the schema.\n\n"
    "The final output must be only the JSON object, with no additional commentary, markdown formatting, or explanatory text."
)

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set your OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    schema = load_schema(SCHEMA_FILE)

    # Tool: Candidate profile parser
    candidate_profile_tool = {
        "type": "function",
        "function": {
            "name": "fill_candidate_profile",
            "description": "Parse the CV and return a candidate profile matching the schema",
            "parameters": schema
        }
    }

    assistant = client.beta.assistants.create(
        name=ASSISTANT_NAME,
        model=MODEL,
        tools=[candidate_profile_tool],
        instructions=INSTRUCTIONS
    )

    print(f"✅ Assistant created: {assistant.name}")
    print(f"🆔 Assistant ID: {assistant.id}")

    with open("assistant_2_id.txt", "w") as f:
        f.write(assistant.id)

    print("📝 Saved Assistant ID to 'assistant_id.txt'")


if __name__ == "__main__":
    main()
