#!/usr/bin/env bash
# ============================================================================
# Shell script validation tests for qwen_audio_sft.sh
#
# Tests the script's command construction, conditional logic, and Hydra
# override wiring without requiring GPU, model weights, or actual training.
#
# Usage:
#   bash tests/test_shell_script.sh
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_UNDER_TEST="${RL2_ROOT}/examples/qwen_audio_sft.sh"

passed=0
failed=0

pass_test() {
    echo "  PASS: $1"
    passed=$((passed + 1))
}

fail_test() {
    echo "  FAIL: $1 - $2"
    failed=$((failed + 1))
}

# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=== Shell Script Syntax and Structure Tests ==="

# Test 1: Script is valid bash syntax
if bash -n "${SCRIPT_UNDER_TEST}" 2>/dev/null; then
    pass_test "Script passes bash syntax check"
else
    fail_test "Script passes bash syntax check" "bash -n failed"
fi

# Test 2: Script has correct shebang
first_line=$(head -1 "${SCRIPT_UNDER_TEST}")
if [[ "${first_line}" == "#!/usr/bin/env bash" ]]; then
    pass_test "Script has correct shebang"
else
    fail_test "Script has correct shebang" "Got: ${first_line}"
fi

# Test 3: Script uses set -euo pipefail
if grep -q "set -euo pipefail" "${SCRIPT_UNDER_TEST}"; then
    pass_test "Script uses strict error handling (set -euo pipefail)"
else
    fail_test "Script uses strict error handling" "set -euo pipefail not found"
fi

# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=== Hydra Override Wiring Tests ==="

# Test 4: Script passes actor.model_name
if grep -q 'actor.model_name=' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Script passes actor.model_name as Hydra override"
else
    fail_test "Script passes actor.model_name" "override not found"
fi

# Test 5: Script passes max_audio_seconds
if grep -q 'data.train.max_audio_seconds=' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Script passes data.train.max_audio_seconds as Hydra override"
else
    fail_test "Script passes data.train.max_audio_seconds" "override not found"
fi

# Test 6: Script passes all three freeze flags
for flag in freeze_audio_encoder freeze_language_model freeze_multi_modal_projector; do
    if grep -q "actor.${flag}=" "${SCRIPT_UNDER_TEST}"; then
        pass_test "Script passes actor.${flag}"
    else
        fail_test "Script passes actor.${flag}" "override not found"
    fi
done

# Test 7: Script passes dataset configuration
for field in dataset_name dataset_subset dataset_split batch_size max_length prompt; do
    if grep -q "data.train.${field}=" "${SCRIPT_UNDER_TEST}"; then
        pass_test "Script passes data.train.${field}"
    else
        fail_test "Script passes data.train.${field}" "override not found"
    fi
done

# Test 8: Script passes trainer configuration
for field in project experiment_name n_epochs; do
    if grep -q "trainer.${field}=" "${SCRIPT_UNDER_TEST}"; then
        pass_test "Script passes trainer.${field}"
    else
        fail_test "Script passes trainer.${field}" "override not found"
    fi
done

# Test 9: Script passes optimizer.lr
if grep -q 'actor.optimizer.lr=' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Script passes actor.optimizer.lr"
else
    fail_test "Script passes actor.optimizer.lr" "override not found"
fi

# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=== Pipeline Stage Tests ==="

# Test 10: Stage 1 - Import skips when checkpoint exists
if grep -q 'if \[ ! -d "${MEGATRON_CKPT_DIR}/iter_0000000" \]' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Stage 1: Import checks for existing checkpoint dir"
else
    fail_test "Stage 1: Import checks" "iter_0000000 check not found"
fi

# Test 11: Stage 1 - Import uses convert_checkpoints.py import
if grep -q 'convert_checkpoints.py import' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Stage 1: Uses convert_checkpoints.py import"
else
    fail_test "Stage 1: Uses convert_checkpoints.py import" "command not found"
fi

# Test 12: Stage 2 - Training uses torchrun/torch.distributed.run
if grep -q 'torch.distributed.run' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Stage 2: Uses torch.distributed.run for training"
else
    fail_test "Stage 2: Uses torch.distributed.run" "command not found"
fi

# Test 13: Stage 2 - Training invokes RL2.trainer.audio_sft
if grep -q 'RL2.trainer.audio_sft' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Stage 2: Invokes RL2.trainer.audio_sft module"
else
    fail_test "Stage 2: Invokes RL2.trainer.audio_sft" "module not found"
fi

# Test 14: Stage 3 - Export checks for saved model
if grep -q 'if \[ -d "${SAVE_DIR}/actor" \]' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Stage 3: Export checks for saved model directory"
else
    fail_test "Stage 3: Export checks for saved model" "check not found"
fi

# Test 15: Stage 3 - Export uses export_hf.py
if grep -q 'export_hf.py' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Stage 3: Uses export_hf.py for HF export"
else
    fail_test "Stage 3: Uses export_hf.py" "command not found"
fi

# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=== Negative Path Tests ==="

# Test 16: Script fails when python_path doesn't exist
if grep -q 'if \[ ! -f "${python_path}" \]' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Negative: Script validates python_path existence"
else
    fail_test "Negative: Script validates python_path" "validation not found"
fi

# Test 17: Script sets PYTHONPATH correctly
if grep -q 'export PYTHONPATH=' "${SCRIPT_UNDER_TEST}"; then
    pass_test "Script exports PYTHONPATH"
else
    fail_test "Script exports PYTHONPATH" "PYTHONPATH not found"
fi

# Test 18: Default environment variables have sensible values
# Extract defaults and validate
hf_model_default=$(grep 'HF_MODEL=' "${SCRIPT_UNDER_TEST}" | head -1 | sed 's/.*:-\(.*\)}/\1/')
if [[ -n "${hf_model_default}" ]]; then
    pass_test "HF_MODEL has a default value: ${hf_model_default}"
else
    fail_test "HF_MODEL has a default value" "no default found"
fi

nproc_default=$(grep 'NPROC=' "${SCRIPT_UNDER_TEST}" | head -1 | sed 's/.*:-\(.*\)}/\1/')
if [[ "${nproc_default}" =~ ^[0-9]+$ ]]; then
    pass_test "NPROC default is numeric: ${nproc_default}"
else
    fail_test "NPROC default is numeric" "got: ${nproc_default}"
fi

max_audio_default=$(grep 'MAX_AUDIO_SECONDS=' "${SCRIPT_UNDER_TEST}" | head -1 | sed 's/.*:-\(.*\)}/\1/')
if [[ "${max_audio_default}" == "30" ]]; then
    pass_test "MAX_AUDIO_SECONDS default is 30"
else
    fail_test "MAX_AUDIO_SECONDS default is 30" "got: ${max_audio_default}"
fi

# ═══════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "Results: ${passed} passed, ${failed} failed, $((passed + failed)) total"
echo "============================================================"

exit ${failed}
