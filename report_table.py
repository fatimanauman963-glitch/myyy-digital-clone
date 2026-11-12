"""Prints a markdown table for the README report: top-3 chunks per mode per question.
Edit QUESTIONS to your own 5 test questions, run `python report_table.py`, then
paste the output into README.md and write the 'best mode and why' column yourself."""
from rag import MODE_BM25, MODE_DENSE, MODE_HYBRID, Retriever

QUESTIONS = [
    "What is the QR Attendance System?",            # exact rare keyword (BM25 should win)
    "Are you an early riser?",                      # paraphrase, shares no words with the answer
    "What do you do for fun besides studying?",     # paraphrase
    "Which technique fuses the two rankings?",      # technical
    "Where do you study?",                          # factual
]

r = Retriever()


def top3(q, mode):
    return ", ".join(h["chunk"]["chunk_id"] for h in r.search(q, mode, 3)) or "(nothing)"


print("| Question | BM25 result | Dense result | Hybrid result | Best mode and why |")
print("|---|---|---|---|---|")
for q in QUESTIONS:
    print(f"| {q} | {top3(q, MODE_BM25)} | {top3(q, MODE_DENSE)} | {top3(q, MODE_HYBRID)} | TODO |")
