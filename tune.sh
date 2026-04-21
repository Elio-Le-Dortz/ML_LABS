#!/bin/bash
# tune.sh — K-Fold CV hyperparameter search for all models
# Does NOT use --test. Use the printed results to pick best hyperparameters,
# then evaluate on test data separately.

DATA="data/features.npz"
N_FOLDS=5
RUN="conda run -n torch python main.py --data_path $DATA --n_folds $N_FOLDS"

# ── Parsing helpers ──────────────────────────────────────────────────────────

acc() { grep -oP 'accuracy = \K[0-9.]+' <<< "$1"; }
f1()  { grep -oP 'F1-score = \K[0-9.]+' <<< "$1"; }
mse() { grep -oP 'MSE = \K[0-9.]+' <<< "$1"; }

run_cv() {
    # run_cv METHOD TASK [EXTRA_ARGS...]
    # Prints a progress note to stderr, returns output to stdout
    local label="$1 / $2 / ${*:3}"
    echo "  running: $label" >&2
    $RUN --method "$1" --task "$2" "${@:3}" 2>/dev/null
}

sep() { printf '%0.s-' {1..68}; echo; }

# ── Header ───────────────────────────────────────────────────────────────────

echo ""
sep
printf "  K-Fold CV Hyperparameter Search   (folds=%s, no test data)\n" "$N_FOLDS"
sep

# ── Linear Regression ────────────────────────────────────────────────────────

echo ""
echo "  LINEAR REGRESSION  [task: regression]"
echo ""
printf "  %-20s  %s\n"  "regularization_param"  "CV MSE"
printf "  %-20s  %s\n"  "--------------------"   "-------"

for reg in 0 0.0001 0.001 0.01 0.1 1 10; do
    out=$(run_cv linear_regression regression --regularization_param "$reg")
    printf "  %-20s  %s\n" "$reg" "$(mse "$out")"
done

# ── Logistic Regression ──────────────────────────────────────────────────────

echo ""
echo "  LOGISTIC REGRESSION  [task: classification]"
echo ""
printf "  %-10s  %-12s  %-12s  %s\n"  "lr"  "max_iters"  "CV Acc (%)"  "CV F1"
printf "  %-10s  %-12s  %-12s  %s\n"  "----------"  "------------"  "------------"  "-------"

for lr in 1e-4 1e-3 1e-2 0.1; do
    for iters in 100 500 1000; do
        out=$(run_cv logistic_regression classification --lr "$lr" --max_iters "$iters")
        printf "  %-10s  %-12s  %-12s  %s\n" "$lr" "$iters" "$(acc "$out")" "$(f1 "$out")"
    done
done

# ── KNN Classification ───────────────────────────────────────────────────────

echo ""
echo "  KNN  [task: classification]"
echo ""
printf "  %-6s  %-12s  %s\n"  "K"  "CV Acc (%)"  "CV F1"
printf "  %-6s  %-12s  %s\n"  "------"  "------------"  "-------"

for k in 1 3 5 7 10 15 20 30; do
    out=$(run_cv knn classification --K "$k")
    printf "  %-6s  %-12s  %s\n" "$k" "$(acc "$out")" "$(f1 "$out")"
done

# ── KNN Regression ───────────────────────────────────────────────────────────

echo ""
echo "  KNN  [task: regression]"
echo ""
printf "  %-6s  %s\n"  "K"  "CV MSE"
printf "  %-6s  %s\n"  "------"  "-------"

for k in 1 3 5 7 10 15 20 30; do
    out=$(run_cv knn regression --K "$k")
    printf "  %-6s  %s\n" "$k" "$(mse "$out")"
done

# ── Footer ───────────────────────────────────────────────────────────────────

echo ""
sep
echo "  Done. Re-run best config with --test to get final test performance."
sep
echo ""
