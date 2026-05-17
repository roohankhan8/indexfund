# KSE-30 Recomposition Notice (Dec 31, 2025) - Coding Guidance

Source PDF: `0-docs/Notice-KSE-30-Index-Recomposition-December2025.pdf`
Notice ID/date in PDF: `PSX/N-142`, dated `January 30, 2026`.

## Recomposition Snapshot
- Recomposition as of: `December 31, 2025`.
- Implementation effective date: `Monday, March 16, 2026`.

## Constituents Changes
Incoming:
- `AIRLINK` - Air Link Communication Limited
- `NRL` - National Refinery Limited

Outgoing:
- Bank AL Habib Limited
- Pakistan Refinery Limited

## Annexure-A Final Constituents (30)
1. AIRLINK
2. ATRL
3. BOP
4. DGKC
5. EFERT
6. ENGROH
7. FCCL
8. FFC
9. GAL
10. GHNI
11. HBL
12. HUBC
13. LUCK
14. MARI
15. MCB
16. MEBL
17. MLCF
18. NBP
19. NRL
20. OGDC
21. PAEL
22. POL
23. PPL
24. PSO
25. SAZEW
26. SEARL
27. SNGP
28. SSGC
29. SYS
30. UBL

## Engineering Use
Use this notice as the authoritative override for the implementation date and incoming/outgoing lists for the Dec-2025 recomposition event.

Recommended data record:
- `index_code`: `KSE30`
- `as_of_date`: `2025-12-31`
- `notice_date`: `2026-01-30`
- `effective_date`: `2026-03-16`
- `incoming_symbols`: `["AIRLINK", "NRL"]`
- `outgoing_companies`: `["Bank AL Habib Limited", "Pakistan Refinery Limited"]`
- `constituents_final`: 30-symbol array from Annexure-A
- `source_notice`: `PSX/N-142`

## Validation
- Ensure final basket size is exactly 30.
- Ensure `AIRLINK` and `NRL` are present in the effective basket.
- Ensure outgoing entities are removed in the effective basket.
- Ensure no composition changes are applied before `2026-03-16`.
