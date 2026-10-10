# Methodology

## Pipeline

```
DATASET -> PREPROCESSING -> TRANSACTIONS -> APRIORI ENGINE -> FREQUENT ITEMSETS -> RULES -> RECOMMENDATION ENGINE -> STREAMLIT UI
data/      preprocessing.py  one-hot bool   apriori_engine.py                      rule_generation.py  recommender.py   app.py
```

## Definitions

With N transactions and count(X) = number of transactions containing every item of X:

- `support(X) = count(X) / N` — fraction of **all** transactions, in [0, 1].
- `support(A -> B) = support(A ∪ B)`.
- `confidence(A -> B) = support(A ∪ B) / support(A)` — fraction of the transactions **containing A**, in [0, 1].
- Antecedent (LHS) = items already in the basket; consequent (RHS) = items the rule suggests.

`A -> B` and `B -> A` share the numerator but are divided by different antecedent supports, so
their confidence generally differs. Confidence is an observed conditional frequency, not a causal
probability.

## Preprocessing (`src/preprocessing.py`)

1. Keep only `transaction_id` and `item`.
2. Drop rows with a missing transaction id (cannot be assigned to a basket).
3. Drop rows with a missing item.
4. Normalise names: trim, collapse spaces, underscores to spaces, Title Case.
5. Drop non-product lines (`Labour Charge`, `Service Fee`, `Discount`).
6. Drop duplicate `(transaction_id, item)` pairs so an item counts once per basket.
7. Group into baskets (sets) and one-hot encode to a boolean matrix; transactions left with no
   product are dropped and counted.

Each step logs rows before/after/changed and a reason (`CleaningReport`).

## Apriori (`src/apriori_engine.py`)

`apriori_native` is written from scratch:

1. Count single items; keep those with `count / N >= min_support` (L1).
2. **Join**: two frequent (k-1)-itemsets (sorted tuples) with the same first k-2 items form a k-candidate.
3. **Prune before counting**: drop the candidate if any (k-1)-subset is not in L(k-1) (Apriori property).
4. **Count**: AND the boolean column of the new item with the stored row-mask of the (k-1)-prefix.
5. **Prune after counting**: keep candidates with support >= min_support (Lk).
6. Repeat until no candidates remain.

`return_trace=True` returns, per level, the candidates joined, those pruned by subset, the counts
and the survivors. The Apriori Learning page and report Section 5 are produced from this trace.

`mine_frequent_itemsets(..., backend="mlxtend")` uses `mlxtend.frequent_patterns.apriori` instead.
`tests/test_apriori.py::test_native_matches_mlxtend` asserts both give identical itemsets and supports.

## Rules (`src/rule_generation.py`)

For each frequent itemset I with |I| >= 2 and each non-empty proper subset A: emit `A -> I \ A`
if `support(I) / support(A) >= min_confidence`. `support(A)` is always available because every
subset of a frequent itemset is frequent. Lift is computed as an extra column but is not used for
thresholds or ranking.

## Recommendations (`src/recommender.py`)

1. Normalise and de-duplicate the basket.
2. Matching rules: antecedent ⊆ basket.
3. Candidates: consequent items not in the basket.
4. Evidence: all matching rules per candidate, sorted by confidence ↓, support ↓, antecedent size ↓.
5. Rank candidates by their best rule (same key), then number of supporting rules, then name.

Result states: `ok`, `empty_basket`, `no_rules` (nothing satisfies the thresholds), `no_match`
(rules exist but none applies). Unknown items are returned in `unknown_items`.

For a rule with a multi-item consequent, the confidence refers to the whole consequent; it is
used as-is (a conservative value) for each item of that consequent.

## Association graph (`src/visualizations.py`)

Nodes are items. Each rule `A -> B` among the top-N by confidence adds a directed edge `a -> b`
for every a in A and b in B. An edge keeps the highest confidence of its contributing rules and
counts them. The graph is therefore item-level; it does not show whole multi-item rules. Each
disconnected cluster is laid out separately so clusters do not overlap.

## Caching (`app.py`)

`st.cache_data` keys: dataset file modification time, minimum support, minimum confidence and the
Apriori backend. Changing any of them recomputes.
