Select up to {{ selected_limit }} truly relevant code context candidates.
Return JSON with keys: selected, user_update, where each selected item has candidate_id, relevance, reason.
user_update should be a short user-facing progress message when useful, or an empty string. Do not reveal chain-of-thought.
Use only candidate_id values from candidates.
Select none if none are relevant.
Do not reject a filename or symbol that directly matches the user's wording only because task_analysis guessed a different programming language.
When task_analysis conflicts with project_profile or concrete repository candidates, prefer repository evidence and select the minimal matching files.

title={{ title }}
description={{ description }}
project_profile={{ project_profile }}
task_analysis={{ task_analysis }}
selected_skills={{ selected_skills }}
queries={{ queries }}
candidates={{ candidates }}
