# StayUp Cricket — Project Submission Note

When you tune into a T20 chase halfway through, broadcasts rarely answer the real question: is this chase still alive? StayUp Cricket gives fans an instant, grounded read without a live score feed.

You type the match state (for example 112/3 13.2 t189) and get the exact equation, both teams’ historical win chances, a verdict on which side history favours, and what comparable IPL chases did over the next over. A second tab replays 15 real IPL chases ball by ball, hiding future scores and the result until you reach them.

Every number comes from Python and Cricsheet IPL data. On 748 held-out chase states from 208 matches (2024 to 2026) the model’s probabilities were more accurate than the required run rate alone (Brier 0.114 vs 0.153).

AI: Groq’s qwen/qwen3.8-27b writes a short explanation from approved facts. Any unsupported number or invented cause is rejected and a calculated explanation is shown instead. Built with Python and Streamlit, assisted by ChatGPT, Google Antigravity/Gemini and Claude.

Limits: past frequency is a guide, never a guarantee. Not affiliated with Dream11, the IPL or Cricsheet.
