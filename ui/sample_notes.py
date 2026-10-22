from pathlib import Path

SAMPLE_NOTES_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "dsa_lecture_notes.pdf"

SAMPLE_QUESTIONS = [
    "Why is appending to a dynamic array O(1) amortised?",
    "What is the difference between BFS and DFS?",
    "When does quicksort hit its worst case?",
    "How do hash tables handle collisions?",
    "What is the difference between memoisation and tabulation?",
]
