# translate/

NLLB-200 distilled 600M (CTranslate2 INT8). It translates Hindi/Tamil questions into English before Llama, and Llama's English answers back.

- `nllb_600m_ct2_int8/`: downloaded by `../download_models.sh` from `JustFrederik/nllb-200-distilled-600M-ct2-int8`. It is not committed. **Licence: CC-BY-NC-4.0 (non-commercial).**
- `nllb_vmap_<lang>/`: built automatically by `lap/server.py` on first use. It contains the per-language vocabulary map (`vmap.txt`) that limits the output vocabulary to the target script. It is not committed.
