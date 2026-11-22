# Demo video script (2–3 minutes)
Record the screen with voice (OBS, Zoom or Xbox Game Bar: Win+G). Keep it under 3:00.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | Title slide | "AI Study Planner: it reads a student's quiz answers and decides what to study next." |
| 0:15–0:40 | Terminal: `python study_planner.py` (scroll to Dana) | "Three students. Dana answered W, P, W, C, C. Bayes gives P(K)=0.058, Viterbi gives B,B,B,R,R." |
| 0:40–1:10 | Same screen, MDP and hill climbing part | "The HMM belief is 0.876 Ready, so the MDP compares Q values: Review 9.03, Next 11.30. Hill climbing picks exercise N1." |
| 1:10–1:50 | `python study_planner.py --ask`: type w w w c c, then c c w w w | "Same answers, different order. First gives Ready 0.86 and Next topic; second gives 0.13 and Review." |
| 1:50–2:20 | Browser: docs/website_en.html | "The same algorithms run in a web version with a lab, games and the evaluation." |
| 2:20–2:45 | Terminal: `python -m unittest discover -s tests -v` | "22 tests pass. Evaluation on 2000 synthetic students: 75.9% against 70.2% and 57.6%." |
| 2:45–3:00 | Last slide | "All parameters are labeled assumptions. Thank you." |
