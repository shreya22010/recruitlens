from flask import Flask, render_template, request, jsonify
from groq import Groq
import json

app = Flask(__name__)
import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    jd = data.get("jd", "")

    prompt = f"""
You are a senior product analyst at an AI-powered hiring platform.

Given this Job Description:
{jd}

Do the following and return ONLY a valid JSON object with exactly these keys: "jd_score", "jd_feedback", "best_fit", "best_fit_reason", "personas"

"jd_score" — a number from 0 to 100 rating how well written the job description is
"jd_feedback" — one sentence on what's good or missing in the job description
"best_fit" — the persona name (string) that is the best overall fit
"best_fit_reason" — one sentence explaining why
"personas" — array of exactly 3 persona objects, each with EXACTLY these keys:
  "persona" — creative name like "The Fresh Grad"
  "background" — 2 sentence description of this candidate type
  "scores" — object with exactly these 4 keys as numbers: "Communication Clarity", "Role Relevance", "Growth Potential", "Culture Fit"
  "ivy_strategy" — one sentence on how to interview this persona
  "questions" — array of exactly 3 interview question strings for this persona

Return ONLY the JSON object. No markdown, no explanation, nothing else.

Example structure:
{{
  "jd_score": 78,
  "jd_feedback": "The JD is clear on skills but lacks information about team size and growth opportunities.",
  "best_fit": "The Excel Expert",
  "best_fit_reason": "Strong role relevance and communication skills make them the safest hire.",
  "personas": [
    {{
      "persona": "The Fresh Grad",
      "background": "Recently graduated with strong academics but limited work experience. Eager to learn and prove themselves.",
      "scores": {{
        "Communication Clarity": 72,
        "Role Relevance": 65,
        "Growth Potential": 88,
        "Culture Fit": 80
      }},
      "ivy_strategy": "Ask situational questions to evaluate potential over experience.",
      "questions": [
        "Tell me about a project where you had to learn a new tool quickly.",
        "How do you prioritize when given multiple tasks with the same deadline?",
        "Describe a time you turned raw data into a useful insight."
      ]
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw = response.choices[0].message.content.strip()

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end != 0:
        raw = raw[start:end]

    result = json.loads(raw)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)