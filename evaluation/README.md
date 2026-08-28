# Conversation Evaluation

Run the fixed Go debugging conversation against an isolated workspace:

```powershell
.\.venv\Scripts\python.exe -m evaluation --case evaluation/cases/go_web_debug.json --config config.json
```

The evaluator records every conversation turn, computes repository changes
without relying on Git, and runs the configured verification command itself.
Artifacts are written under `.repomind/evaluation/<case_id>/<run_id>/`.
