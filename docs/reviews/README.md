# Independent reviews

FinSight was scored in 14 rounds by an independent reviewer agent that had not built the project, against the rubric in
[rubric.md](rubric.md) (13 dimensions, 0–100 each). Each review re-ran the checks itself, probed the system with its own
unseen questions, claims and planted documents, listed bugs with repros, and wrote a hidden held-out slice for the next
round (committed afterwards as `evaluation/heldout_r<N>/`, see [docs/evaluation-history.md](../evaluation-history.md)).
The fixes between reviews were made by separate engineer agents; scores are the reviewer's, not the authors'.

| Round | Commit reviewed | Overall | Lowest dimension | Report |
|---|---|---|---|---|
| 1 | `–` | 77 | 68 | [round1.md](round1.md) · [json](round1.json) |
| 2 | `d5d6a0d` | 82 | 70 | [round2.md](round2.md) · [json](round2.json) |
| 3 | `4742453` | 86 | 75 | [round3.md](round3.md) · [json](round3.json) |
| 4 | `fbce040` | 89 | 82 | [round4.md](round4.md) · [json](round4.json) |
| 5 | `a125300` | 90 | 83 | [round5.md](round5.md) · [json](round5.json) |
| 6 | `bc42017` | 91 | 84 | [round6.md](round6.md) · [json](round6.json) |
| 7 | `20e57a0` | 93 | 87 | [round7.md](round7.md) · [json](round7.json) |
| 8 | `40e8685` | 93 | 86 | [round8.md](round8.md) · [json](round8.json) |
| 9 | `cbcf3b2` | 93 | 86 | [round9.md](round9.md) · [json](round9.json) |
| 10 | `7f4f06c` | 93 | 87 | [round10.md](round10.md) · [json](round10.json) |
| 11 | `4ad3380` | 94 | 87 | [round11.md](round11.md) · [json](round11.json) |
| 12 | `bb99360` | 93.8 | 88 | [round12.md](round12.md) · [json](round12.json) |
| 13 | `aa73005` | 94.2 | 89 | [round13.md](round13.md) · [json](round13.json) |
| 14 | `64cfbeb` | 94.5 | 90 | [round14.md](round14.md) · [json](round14.json) |

Round 1 predates the per-commit review format. The target set by the owner was every dimension at 90 or above, reached
in round 14. Items the reviews list as owner-blocked (human answer labels, a 问财/豆包/Kimi head-to-head, owner-collected
real claims, a user study) are not done; the kit is in `evaluation/human/`.
