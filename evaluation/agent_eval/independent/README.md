# Scripts of the independent evaluation sets

These scripts were written by the authors of the independent sets (`tasks/agent_eval_multiturn_v1.jsonl`,
`tasks/agent_eval_test_v3.jsonl`, `tasks/router_labels_independent_v1.jsonl`) outside the repository and copied
here unchanged except for paths. The `build_*` scripts generated the sets; the `verify_*` scripts re-run the offline
tools and check every expected fact. See `tasks/README_multiturn_v1.md` and `tasks/README_test_v3.md` for the
construction protocols. They are excluded from lint so they stay as the authors wrote them.

    PYTHONPATH=. python evaluation/agent_eval/independent/verify_test_v3.py
    PYTHONPATH=. python evaluation/agent_eval/independent/verify_multiturn.py
