# Task And Candidates

title={{ title }}
description={{ description }}
project_profile={{ project_profile }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
queries={{ queries }}
candidates={{ candidates }}

# Selection Rules

Select up to {{ selected_limit }} truly relevant code context candidates.
Use only candidate_id values from candidates, and select none when none are relevant.
Do not reject a filename or symbol that directly matches the user's wording only because task_analysis guessed another language.
When task_analysis conflicts with project_profile or concrete candidates, prefer repository evidence and select the smallest relevant set.

# Output Schema

Return JSON with keys: selected, user_update. Each selected item must contain candidate_id, relevance, and reason.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
