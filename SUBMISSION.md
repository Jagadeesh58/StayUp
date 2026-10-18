# Short Note: StayUp Cricket

Maximum 200 words

- 🌐 **Live Web Application**: [https://stayup.streamlit.app/](https://stayup.streamlit.app/)
- 📦 **GitHub Repository**: [https://github.com/Jagadeesh58/StayUp](https://github.com/Jagadeesh58/StayUp)

### The fan problem
Fans often tune into a T20 chase halfway through. The broadcast shows the score but rarely explains what it means. Is 112/3 after 13.2 overs, chasing 189, comfortable or desperate?

### How the solution helps
StayUp Cricket answers one question: "Is this chase still alive?" The fan types the score, for example 112/3 13.2 t189, and instantly sees the exact equation, both teams' historical win chances, a plain verdict, and the typical runs similar IPL chases scored in the next over, plus how often a wicket or boundary came. A spoiler-free replay lets fans step through 15 real chases ball by ball. Every number is calculated in Python from 1,172 IPL matches. On 208 unseen matches from 2024 to 2026, it beat a required-run-rate baseline (Brier 0.114 vs 0.152).

### AI tools used
In the app, a Groq-hosted Qwen model writes short explanations in seven languages, including Hindi, Telugu and Tamil. It only uses facts supplied by Python, and any unsupported number is rejected and replaced by a calculated explanation. Used to build the project: ChatGPT, Google Antigravity/Gemini and Claude.
