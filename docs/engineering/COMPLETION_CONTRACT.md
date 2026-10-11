# Capability completion contract

A capability is **COMPLETE** only when all of the following hold:

1. Implementation is wired into the **production** workflow (not a side prototype).
2. Governing equations, BCs, and supported operating range are documented.
3. Unit and integration tests pass with **documented** tolerances (not ad-hoc loosening).
4. Independent analytical or MMS verification exists where applicable.
5. Conservation / refinement evidence is recorded where applicable.
6. Public API and E2E entry points work without hidden defaults.
7. Capability registry and verification matrix match reality.
8. Full regression suite passes except explicitly justified skips.
9. No **critical** open defect for that capability in `DEFECT_REGISTER.md`.

Statuses used in `MASTER_ROADMAP.md` and `capabilities.py`:

| Status | Meaning |
|--------|---------|
| **COMPLETE** | All criteria above satisfied. |
| **IN PROGRESS** | Production path exists; acceptance criteria partially met. |
| **BLOCKED** | Specific dependency or environment prevents closure. |
| **PLANNED** | Not started or not in production path. |

Passing pytest alone does **not** imply COMPLETE.
