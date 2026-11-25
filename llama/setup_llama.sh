#!/usr/bin/env bash
# Download Meta's Llama 3.2 1B Instruct QLoRA INT4 checkpoint and export it to ExecuTorch (XNNPACK, 8da4w).
# Needs: `hf auth login` done, and the licence accepted at
#   https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/ps2_et/bin:$PATH"   # export needs the venv's flatc on PATH
PY="env -u PYTHONPATH $HOME/ps2_et/bin/python"
HF="env -u PYTHONPATH $HOME/ps2_et/bin/hf"
CKPT=checkpoint
$HF download meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8 \
    consolidated.00.pth params.json tokenizer.model --local-dir "$CKPT"
CFG=$($PY -c "import executorch;print(list(executorch.__path__)[0]+'/examples/models/llama/config/llama_xnnpack_qat.yaml')")
$PY -m executorch.extension.llm.export.export_llm \
    --config "$CFG" \
    +base.model_class=llama3_2 \
    +base.checkpoint="$CKPT/consolidated.00.pth" \
    +base.params="$CKPT/params.json" \
    +export.output_name=llama3_2_1b_qlora_int4_xnnpack.pte
ls -lh llama3_2_1b_qlora_int4_xnnpack.pte
