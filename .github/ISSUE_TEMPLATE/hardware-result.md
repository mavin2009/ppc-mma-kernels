---
name: Hardware validation result
about: Report a temperature-0 check + llama-bench run from real Power10/11 silicon
title: "[hardware] <machine> / <model> / <PASS or FAIL>"
labels: hardware-result
---

Paste the contents of `validation-report.md` (produced by
`./scripts/validate-on-power.sh /path/to/model.gguf`) below.

If the temperature-0 check FAILED, please also attach
`/tmp/out-mma.txt` and `/tmp/out-ref.txt` — a divergence is exactly
the data this project most needs.

<!-- report below this line -->
