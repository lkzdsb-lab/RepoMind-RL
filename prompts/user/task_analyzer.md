Return JSON with keys: task_type, verification_required, verification_reason, task_category, entities, acceptance_criteria, risk_notes, search_hints.
task_type must be one of BUG_FIX, FEATURE_IMPL, DIAGNOSE.
verification_required must be a boolean decided from the user's intent.
Set verification_required=true when running the configured verification command is necessary for the requested outcome.
Set verification_required=false when the requested outcome can be satisfied by reading, analyzing, or reporting issues without command-based verification.
verification_reason should briefly explain that decision.
entities and search_hints should come from the user's actual task wording only.
If a field is uncertain, use an empty list/string instead of guessing.
First inspect project_profile to understand repository language evidence, then combine it with the user's natural language.
Do not infer a programming language from the word "main" alone. It can refer to Python, Go, Java, Rust, C/C++, or a filename.
Keep language-specific criteria out unless project_profile or the user explicitly supports that language.
If project_profile has one clear primary_language and the user uses language-neutral terms, prefer that repository language.

title={{ title }}
description={{ description }}
current_task_type={{ current_task_type }}
verify_command={{ verify_command }}
project_profile={{ project_profile }}
registry_snapshot={{ registry_snapshot }}
