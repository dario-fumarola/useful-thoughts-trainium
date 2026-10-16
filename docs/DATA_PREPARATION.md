# Mathematical corpus preparation

The surviving original corpus contains token smoke shards, not the audited raw
documents required for one document-level anchor. The new corpus therefore uses
the declared FineMath-4+ fallback, without changing the mathematical-data recipe.
All downloaded documents, benchmark sources, questions, records, and detailed
manifests remain under ignored `local/data/`.

## Pinned sources and permitted use

| Source | Revision | Role |
|---|---|---|
| `HuggingFaceTB/finemath`, `finemath-4plus` | `e92b25a616738fe95dc186b64dfb19f9c8525594` | Raw mathematical documents |
| `openai/gsm8k`, `main` train/test | `740312add88f781978c0658806c59bc2815b9866` | Question-only corpus exclusion |
| `allenai/ai2_arc`, `ARC-Challenge` all splits | `210d026faf9955653af8916fad021475a3f00453` | Question-only corpus exclusion |
| `EleutherAI/lambada_openai`, English test | `900124bf3b8235c6daf21033af9948b3f07346c4` | Prefix-only corpus exclusion; final target word removed |
| `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Local tokenizer |

FineMath is attributed to HuggingFaceTB and the FineMath/SmolLM2 data authors.
Its pinned [dataset card](https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md)
specifies ODC-By 1.0 and the Common Crawl terms. The card reports upstream
MinHash deduplication and test-set 13-gram exclusion; the local audits below
provide an additional screen. Source cards and file inventories are saved in
`local/data/sources/`. The benchmark projection reads only named question/text
Parquet columns; answer/answer-key columns never become Python rows. It creates
no scores, model outcomes, or prompt-selection evidence.

## Selection and audit

`scripts/fetch_corpus.py` resolves the immutable revision before streaming. The
source slice uses seed 1234 and a 10,000-document shuffle buffer. It is a shuffled
stream slice, not an exactly uniform sample of all FineMath documents. Production
raw rows retain source URL and WARC/crawl/score metadata supplied upstream.

`scripts/prepare_audited_corpus.py` applies these fixed steps:

1. Tokenize locally without adding special tokens. Mark literal tokenizer special
   IDs, and count documents shorter than 512 prefix + 16 future tokens.
2. Screen lowercase normalized 13-word overlaps with 16,535 benchmark
   questions/prefixes. Screen 686 shorter questions in full; seven questions
   below five words are reported but have no additional short-question screen.
3. Join normalized-whitespace exact copies, versions of the same URL, and
   verified near duplicates. Document identity preserves sorted content-bearing
   query parameters, removing only named trackers (`utm_*`, `gclid`, `fbclid`,
   `msclkid`, `dclid`, `mc_cid`, `mc_eid`). Near-duplicate candidates use
   64-permutation MinHash, 16 bands of four, and five-word shingles. Every proposed
   edge is verified with exact shingle-set Jaccard >= 0.8.
4. Form connected components before splitting. Exclude a whole component when
   any member overlaps a benchmark or contains a special token. Retain its first
   source document long enough for a record.
5. Reuse `make_record` for one hash-selected anchor and document-hash split;
   preserve the 512/16 lengths. Hash-sort records before writing. C and R must use
   the same `seed_order` permutation from `weekend_thinking/order.py`.

MinHash candidate discovery is probabilistic. Neither that audit nor normalized
word overlap proves absence of every paraphrase. These checks make no claim
about benchmark contamination in released pretrained weights. Selection changes
the distribution and explicitly conditions on document length.

## Evidence and commands

The 5,000-document pilot produced 3,535 records: 3,180 train, 179 dev, and 176
test. There were 1,265 short documents and 200 redundant eligible page versions;
no additional verified near duplicates, benchmark overlaps, or special-token
documents were found. Its output was byte-identical to the preparation helper
before the URL identity correction on the same pilot: SHA-256
`84eb6dcfe29528dd5de57c43904ba50cfd16da34c65d3068540808ae1bfeb8c1`.
The pilot is engineering evidence, not the full experiment corpus.

Production download contains 500,000 raw documents. A 400,000-document slice
would be marginal for the maximum 262,144 fresh training records, using the
pilot's eligibility rate. The production audit is run with:

```bash
source scripts/env.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/prepare_audited_corpus.py \
  --documents local/data/finemath/production-500000.raw.jsonl \
  --benchmark-questions local/data/benchmark_exclusion/questions.jsonl \
  --out local/data/finemath/production-500000.audit-v2.records.jsonl \
  --tokenizer-dir local/assets/qwen3-1.7b-base \
  --tokenizer-identity Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --source-revision e92b25a616738fe95dc186b64dfb19f9c8525594 --workers 8
```

Do not overwrite or silently append to an audited record file. Its sibling
`.manifest.json` binds raw input, exclusion questions, output, and document
membership hashes. `local/data-production-audit-v2.log` records progress and the
final aggregate receipt.

V2 produced **366,134 records: 329,535 train, 18,360 dev, 18,239 test**. That
supports 2,574 complete fresh batch-128 updates, exceeding the 2,048-step budget
candidate. Its record SHA-256 is
`bd81972fff861f497ce61479079d6904e1baef0a335dfc848cc33e4d90d069ff`.
The audit found 6,267 verified near-duplicate edges and 105 directly overlapping
benchmark documents (66 GSM8K, 39 ARC). Whole-component exclusion removed 106
benchmark-connected documents and one special-token document. There were
127,531 short raw documents; 6,236 redundant eligible documents were removed.

The first production audit is retained at
`local/data/finemath/production-500000.audited.records*` as evidence of a detected
pre-freeze bug. The old helper discarded all query parameters, conflating 9,210
different `jiskha.com/display.cgi?id=...` questions, 2,079 distinct forum threads,
and 1,586 distinct AoPS wiki pages, among others. That audit produced 302,100
train records and excluded 12,775 documents in 89 benchmark-matched components.
`local/data/finemath/url_group_diagnostic.json` records the aggregate URL
diagnostic. V2 corrects identity before the experiment is frozen; v1 is not the
scientific training asset. The raw source, tokenizer, lengths, deterministic
anchor/split algorithm, and objective are unchanged.

After preparation, verify input/source/question/tokenizer hashes, every retained
record, and 48 sampled raw-source anchors with:

```bash
python scripts/verify_corpus_assets.py \
  --records local/data/finemath/production-500000.audit-v2.records.jsonl \
  --documents local/data/finemath/production-500000.raw.jsonl \
  --benchmark-questions local/data/benchmark_exclusion/questions.jsonl \
  --model-dir local/assets/qwen3-1.7b-base \
  --out validation/data_production_v2_20261016.json
```

The resulting [verification receipt](../validation/data_production_v2_20261016.json)
passed: all 366,134 records met identity, split, length, vocabulary, special-token,
and uniqueness checks; all 48 sampled raw-source anchors reconstructed exactly.
It also binds tokenizer files and records the existing `seed_order` algorithm's
complete training-pool hashes for seeds 1234 and 5678 and the first 512 dev /
2,048 final local-document ID hashes. These IDs use fixed file order; they do not
inspect model outcomes or choose a checkpoint.

Regression coverage includes exact normalized question matching, full short
questions, verified near-duplicate discovery, transitive identity, final LAMBADA
target removal, and exclusion propagation across page versions and near copies.
The initial full CPU check passed 89 tests; logs are in
`local/checks/tests.7tNzJu/`. After the URL identity correction, all 16 data/CLI
tests passed. A subsequent complete-suite run exposed and reported an unrelated
import-time XLA/GRU patch in new accelerator modules; that repair and full rerun
are coordinated by the implementation agents. Data checks provide no Neuron
qualification evidence.
