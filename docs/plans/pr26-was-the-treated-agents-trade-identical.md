# PR26 · Was the treated agent's trade identical across arms in the injected design?

- Written 2026-09-07, before any run existed. **Frozen on commit.**
- Governs E51. Reruns the PR5 design (E27) with one new measurement. Does not modify, pool with, or
  supersede E27's numbers, which stay as published with their 2026-09-07 correction.

## The question, in plain words

In the injected design the treated agent reads its arm's rendering of the report, decides and
trades, and only then has its outgoing message replaced by a fixed text. E27 and the submitted
paper said its trade was therefore identical across arms and so was the price its teammates saw a
round later. That was reasoning, not measurement: the agent read different texts before deciding,
and nothing compared its trades. E27's run directories were deleted by a `/tmp` cleanup, so the
question can only be answered by running the design again. **How often did the treated agent trade
the same way in both arms, and how often was the price its teammates saw identical?**

## Why it is worth the GPU time

The teammate-transmission result is one of the paper's contributions, and the design was described
as holding the price fixed. A count settles whether it did.

## Design: an exact rerun of PR5

288 runs: 4 scenarios x 36 seeds (300 to 335) x 2 arms, the configs in `configs/experiments/pr5/`
as committed, roster `configs/models/qwen3_4b_shared.yaml` (Qwen3-4B-Instruct-2507 at the pinned
revision), 12 rounds, injection at round 8, topology full. Outputs go under `outputs/runs/e51/`,
never a scratch directory. Generation is intermittently non-reproducible, so this is a fresh
replication of the design, not a reconstruction of E27's data.

## Outcomes, fixed now

Unit: one matched pair (a seed and scenario, neutral and affective arms).

1. **Primary: identical trade.** The treated agent's filled order at the shock round (round 8) is
   identical across arms: same `position_after` at round 8 and at round 7. Reported as a count over
   pairs, per scenario and pooled, alongside the count of pairs whose decision `action` and `size`
   were identical (the requested trade, before the position limit).
2. **Primary: identical observed price.** The market price after round 8 clears (the price every
   agent observes at round 9) is identical across arms. Count over pairs, per scenario and pooled.
3. **Gate status.** `check_pair` for every pair, reported as in E27. Counts 1 and 2 are reported
   over all pairs and over gate-passing pairs.
4. **Secondary, descriptive: peer response replication.** The registered E27 estimator
   (`affective_contagion.verification.peer_response`: sum over the five untreated agents of anxious
   minus calm round-9 arousal, scrubbed, signed lexicon, bootstrap interval) on the rerun. Reported
   beside E27's `+0.0895` `[+0.0760, +0.1028]`, never pooled with it.

## Decision rule

There is no threshold. The counts are the result. The paper's sentence becomes: "In a rerun of
the design after submission, the treated agent's trade was identical in k of n gate-passing pairs
and the price its teammates saw was identical in m of n." If k = n the design's intent held in this
replication; if k < n the paper says so and the injected result is read as one message plus a
price difference in n minus k pairs. Outcome 4 is reported whatever it shows.

## Exclusions and failure

- A run with no `records.jsonl` is a failed run; the queue stops and the failure is reported. The
  entry records completed and failed counts.
- Pairs failing `check_pair` are counted and reported, not dropped.
- Parser failures fall back to HOLD at size 0 as the simulator defines; the count of fallbacks is
  reported.
- If fewer than 288 runs complete, the entry is partial and says so; no count is reported as a
  population statement.

## What the result can and cannot support

It can say how often the price channel was closed in this replication of the design. It cannot say
what happened in E27's own runs, which are gone. It does not change any E27 number.

## Analysis

`scripts/e51_trade_identity.py`, hashed and committed with this plan before the queue starts; its
known-answer test is `tests/test_e51_trade_identity.py`. Outcome 4 uses the existing registered
estimator unchanged.

## Hashes at freeze (SHA-256, first 16 hex)

| File | Hash |
| --- | --- |
| `src/affective_contagion/trade_identity.py` | `117c586b1aa810f7` |
| `scripts/e51_trade_identity.py` | `8c997a5ef5463217` |
| `scripts/e51_queue.sh` | `5e55fde7563d68cc` |

The known-answer test was shown to fail on a comparator that ignored the requested size before the
correct version was committed.

## Amendment A, 2026-09-07 (queue running; no E51 output examined)

Written while the second run of 288 was generating and before `scripts/e51_trade_identity.py` or
any estimator had been run on E51 data. While checking that the registered E27 estimator could read
the E51 run names, we found that it excludes the sentiment analyst in every pair although PR5
balanced the treated role across the six roles (E27, second correction of 2026-09-07). Outcome 4 is
therefore split, and one outcome is added:

- **4a.** The registered route, `peer_response` (fixed sentiment analyst), reported only so the
  rerun can be compared with E27's published figures.
- **4b.** `peer_response_by_manifest`, which excludes the treated role named in each pair's
  manifest. This is the estimate of the five-teammate response for this rerun, with and without
  echo-dropping.
- **5.** `treated_self_response`: the treated agent's own round-9 shift, so the reader can see what
  route 4a was adding in place of a teammate.

All three are descriptive replication outcomes and are never pooled with E27. Hash of
`src/affective_contagion/verification.py` at this amendment: `9ad78e5602565d93`. The new tests show the fixed-role
route disagreeing with the manifest route on a pair whose treated role is the trader.

## Amendment B, 2026-09-07 (queue running; no E51 output examined)

`scripts/e51_replication.py` (SHA-256 `769dae67a8360cf3`) is the runner for outcomes 4a, 4b and 5. It only
calls the estimators hashed in Amendment A, at round 9, and writes them as JSON.

## Amendment C, 2026-09-07 (queue at run 24 of 288; no E51 output examined)

The audit of 2026-09-07 found that E38's injected per-role table and E43's addendum (the teammate
discriminant) inherit E27's sender error. E51 therefore also reports, descriptively and never pooled
with the originals:

- **6.** The per-role teammate table by manifest: each of the six roles over the pairs in which it
  was not the sender (120 of 144 when all pairs complete), paired arousal shift at round 9 with a
  percentile bootstrap interval and the count of positive pairs.
- **7.** E43's discriminant ((fear + nervousness) minus (excitement + surprise + anger + annoyance)
  from the pinned GoEmotions checkpoint), anxious minus calm at round 9, summed over the five
  untreated teammates named by each pair's manifest and divided by five, with and without dropping
  teammates whose anxious-arm message contains an affect-naming term.

Runner: `scripts/e51_teammates.py`, SHA-256 `182ea6f7f404b50b`. The classifier runs on CPU so it does not
compete with the queue.

## Deviation note, 2026-09-07 (queue complete; before any analysis ran)

`src/affective_contagion/verification.py` hashes `23a989d11b747c9f` at analysis time, not the value in Amendment A,
because the E19 correction added `exact_spearman_p(one_sided=...)`, `LADDER_EFFECTS` and
`spearman_rho` to the same module after the amendment. Every function E51's runners call
(`peer_response`, `peer_response_by_manifest`, `treated_self_response`, `_peer_response`,
`_manifest_treated_role`, `_arousal`, `_bootstrap`, `matched_pairs`, `_teammate_messages`) is
byte-identical to the version at commit `a208d9f`, checked function by function before this note was
written. The other five hashes match.

## Amendment D, 2026-09-07 (after the frozen runners had been executed once)

A pre-analysis check found that `e51_replication.py`
loads the lexicon with exact surface matching while the registered E27 route and `e51_teammates.py`
match word endings too (`surface_then_suffix`). The frozen run's outputs are kept as written
(`outputs/e51_replication.json`). The runner gained a `--vad-match` option, default `surface` so the
frozen behaviour is unchanged (new SHA-256 `b6f854be0861c280`), and was run again with
`surface_then_suffix` (`outputs/e51_replication_suffix.json`). Both are reported; the registered
matching is the one comparable with E27's figures. The independent verifier
(`affective_contagion.injected_verification`) computes both modes from the released archive as a
second route.
