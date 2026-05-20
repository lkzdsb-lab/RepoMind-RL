Return JSON with keys: task_type, verification_required, verification_reason, task_category, entities, acceptance_criteria, risk_notes, search_hints.
task_type must be one of BUG_FIX, FEATURE_IMPL, DIAGNOSE.
verification_required must be a boolean decided from the user's intent.
Set verification_required=true when running the configured verification command is necessary for the requested outcome.
Set verification_required=false when the requested outcome can be satisfied by reading, analyzing, or reporting issues without command-based verification.
verification_reason should briefly explain that decision.
entities and search_hints should come from the user's actual task wording only.
If a field is uncertain, use an empty list/string instead of guessing.

title={{ title }}
description={{ description }}
current_task_type={{ current_task_type }}
verify_command={{ verify_command }}
registry_snapshot={{ registry_snapshot }}
