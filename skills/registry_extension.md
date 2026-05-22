# Registry Extension

Use this skill when adding runtime tools, nodes, prompts, or skills.

## Registry Model

- `RegistryManager` owns mutable registries.
- `RegistrySnapshot` freezes tools, nodes, prompts, and skills for one agent run.
- `ManifestLoader` loads JSON/TOML manifests into the mutable registries.

## Tool Extension

1. Define a `ToolSpec` with name, description, runner, schema, permissions, and optional reducer.
2. Runner signature should accept `(repo_path, args)` and return a dictionary.
3. Reducer should update `AgentState` with structured outputs only.
4. Keep tool output contracts stable because policy, reward, memory, and context layers consume them.

## Prompt Extension

1. Put system and user templates under `prompts/system` and `prompts/user`.
2. Use `prompts.templates.render_prompt()` for runtime rendering.
3. Keep business prompt text out of execution classes.

## Skill Extension

1. Add the reusable workflow as a resource under `skills/`.
2. Register a `SkillSpec` with accurate triggers and resources.
3. Verify skill retrieval by checking `registry_snapshot.skills`, `skill_context`, and `selected_skills`.

## Avoid

- Do not mutate a run's active snapshot after the run starts.
- Do not add tools without reducers when state updates are required.
- Do not put large prompt text directly in Python modules.
