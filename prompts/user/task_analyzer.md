Return JSON with keys: task_type, task_category, entities, acceptance_criteria, risk_notes, search_hints.
task_type must be one of BUG_FIX, FEATURE_IMPL, DIAGNOSE.
entities and search_hints should come from the user's actual task wording only.
If a field is uncertain, use an empty list/string instead of guessing.

title={{ title }}
description={{ description }}
current_task_type={{ current_task_type }}
review_only={{ review_only }}
verify_command={{ verify_command }}
registry_snapshot={{ registry_snapshot }}
