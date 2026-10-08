# llm_blender_addon — Installation Guide

A Blender addon that adds a chat panel backed by a local LLM running
**fully locally** (via Ollama), with automatic model recommendation based
on your available VRAM and one-click download from Hugging Face. Qwen is
one supported model family; the addon also supports other LLMs published
on Hugging Face, including Llama, Gemma, Phi, Mistral, SmolLM, TinyLlama,
GLM, DeepSeek, and Kimi models available as compatible GGUF files.

This guide assumes nothing is installed yet.

---

## 1. Prerequisites

| Tool | Why | Link |
|---|---|---|
| Blender 4.1 or newer | Runs the addon | [blender.org](https://www.blender.org/download/) |
| Ollama | Serves the selected LLM locally on `localhost:11434` | [ollama.com](https://ollama.com/download) |
| Python `huggingface_hub` | Optional: enables resumable/robust downloads | installable from inside the addon (step 4) |

A GPU isn't strictly required (Ollama can run on CPU), but without one,
models above 4-8B will be slow.

---

## 2. Install Ollama

**Windows**
1. Download the installer from [ollama.com/download](https://ollama.com/download).
2. Run it and follow the setup wizard.
3. Ollama starts automatically in the background (system tray icon) and
   listens on `http://localhost:11434`.

**macOS**
```bash
brew install ollama
```
or download the app directly from [ollama.com/download](https://ollama.com/download).

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
This installs Ollama as a systemd service, started automatically.

**Verify it's running** (all OSes), in a terminal:
```bash
ollama -v
```
If a version number prints, you're good. You can also open
`http://localhost:11434` in a browser — it should reply
`Ollama is running`.

---

## 3. Install the addon in Blender

1. Grab the `llm_blender_addon.py` file.
2. In Blender: **Edit > Preferences > Add-ons**.
3. Click the small dropdown arrow in the top-right corner of the window
   (next to the add-ons list) → **Install from Disk...**
   *(on recent Blender versions, "Install from Disk" handles both
   `.zip` extension packages and legacy single-file `.py` add-ons — this
   is the second case here.)*
4. Select `llm_blender_addon.py` and confirm.
5. Tick the checkbox next to **"Local LLM Assistant"** in the list to
   enable it.

The panel shows up in the **3D Viewport**: press **N** to open the
sidebar, then the **"Local LLM"** tab.

---

## 4. Install `huggingface_hub` (recommended, optional)

The addon works without it — the model scan already queries the Hugging
Face API directly for real file sizes and filenames. What
`huggingface_hub` adds is a more robust, resumable download (useful on
flaky connections or for very large files).

In the **Local LLM** panel, if `huggingface_hub` isn't detected, an
**"Install huggingface_hub"** button appears above the scan button —
click it to install it directly into Blender's Python.

If that button fails (permissions, proxy, etc.), install it manually
from a terminal, targeting **Blender's own Python** (not your system
one):

**Windows** (adjust the path to your Blender version):
```powershell
"C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe" -m pip install huggingface_hub
```

**macOS**:
```bash
/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install huggingface_hub
```

**Linux**:
```bash
/path/to/blender/4.2/python/bin/python3.11 -m pip install huggingface_hub
```

Restart Blender afterwards.

---

## 5. First run: pick and fetch a model

1. Open the **Local LLM** panel (N-sidebar, 3D Viewport).
2. Set the **VRAM budget** slider to how much you're willing to dedicate
   to the model (e.g. 8 for an 8-12 GB card, leaving headroom for
   Blender itself).
3. Click **"Scan for available models"**. The addon queries Hugging Face
   and lists compatible Hugging Face models that fit that budget, from the
   largest/highest-quality down to the most compact. Qwen is only one of
   the available model families. Hover over any model's name (the ⓘ icon)
   to see a short description of what it is and what it's good for before
   downloading anything.
4. Next to the model you want, click the **download** icon (arrow). The
   `.gguf` file is saved into the folder set in **"Models folder"**
   (defaults to `~/llm_blender_models`). This can take a few minutes
   depending on size and connection speed — Blender's UI stays usable
   meanwhile.
5. Once downloaded, click the **✓** icon that replaces it to
   **register it in Ollama** — this creates the corresponding Ollama
   model (equivalent to running `ollama create`).
6. The **active model** field at the bottom updates with the registered
   model's name — that's what the chat will use.

*Note: you can also skip the scan/download flow entirely and use the
regular `ollama pull` command line, then type the model name into that
field — the buttons are a convenience, not a requirement.

**Browsing manually by family**: instead of the automatic scan, the
**"Browse by family"** box lets you pick a family from a dropdown (Qwen,
Llama, Gemma, Phi, Mistral, SmolLM, TinyLlama, GLM, DeepSeek, Kimi...),
then a specific model from a second dropdown — each entry's tooltip
already shows a short description. Click **"Check available sizes"** to
list every quantization Hugging Face has for that specific model,
regardless of your VRAM slider, then download/register any of them the
same way as above.

---

## 6. Using the chat

At the bottom of the panel: a message field and a **Send** button. The
model's reply appears in the history box above it.

Two checkboxes at the top change the addon's behavior:

- **Simple assistant** (checked by default): the AI answers questions,
  explains things, suggests code snippets — nothing is ever executed
  automatically.
- **Agentic control**: the AI is allowed to propose `bpy` Python code to
  act on the scene. If a code block appears in its reply, a
  **"Proposed code"** box shows up with an **"Execute proposed code"**
  button.

  ⚠️ **That button runs code with the same permissions as Blender
  itself.** Always read the code before clicking — nothing runs without
  that explicit click, but nothing stops you from clicking without
  reading either. Save your `.blend` file first if it matters.

---

## 7. 3D generation (image / text → mesh)

The **3D generation** box runs [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B)
(MIT) fully locally through [trellis.cpp](https://github.com/pwilkin/trellis.cpp),
a C++/ggml runtime with GGUF weights and a resident HTTP server — no Python or
PyTorch needed at run time. It is **not** an Ollama model: Ollama only runs
text LLMs, so 3D generation uses its own server.

Follow the numbered steps in the panel:

1. **Runtime** — runs the project's official installer in non-interactive mode
   (runtime only, a few tens of MB; no weights, no desktop app). Supported on
   Linux and Windows. On macOS, get/build trellis.cpp yourself and fill in
   *trellis-server binary*.
2. **Weights** — the *3D VRAM* slider picks the quantization tier: **f16**
   (~16.5 GB), **q8** (~9.5 GB) or **q4** (~6 GB). The 10 GGUF files are
   downloaded with a progress bar, speed and remaining time. Downloads are
   resumable and can be cancelled; re-running skips files already complete.
3. **Server** — *Start* launches `trellis-server` (default `127.0.0.1:8080`)
   with the downloaded weights; the first start loads the pipeline onto the
   GPU and can take a moment. The log is kept in `<3D folder>/logs/`.
4. **Source** — an image, or a text prompt. Text mode is a two-step pipeline:
   Z-Image (via [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp))
   makes the image, then TRELLIS.2 makes the mesh. You must provide the
   `sd-cli` binary yourself (build it or take a release), then use
   *Download Z-Image weights* (diffusion + text encoder + VAE).
5. **Generate** — sends the image to the server and imports the resulting
   `.glb` into the scene. The per-stage progress (`[k/6]`) comes from the
   server log. Published timings: roughly 3–7 min on a dedicated GPU,
   6–13 min on an iGPU.

---

## 8. Troubleshooting

**"Ollama unreachable" in the chat**
Check that Ollama is actually running (`ollama -v` in a terminal, or the
tray icon on Windows). If you changed the default port, update
`OLLAMA_URL` near the top of the addon file to match.

**Scan finds no models**
Your VRAM budget may be too low for the catalog's models — try raising
the slider. Below ~1.5-2 GB, even the smallest available models may not
fit once the context-overhead margin is subtracted.

**Download fails with a 404 error**
This was a bug in early versions of this addon: without
`huggingface_hub` installed, it used to *guess* a plausible filename
instead of checking the real one, and that guess didn't always match
what's actually in the Hugging Face repo. Current versions query
Hugging Face's public file listing directly during the scan, so the
filename shown is always the real one — if you still see a 404, it's an
edge case (network hiccup between scan and download, or a renamed repo)
rather than the norm; rerun the scan right before downloading and check
your connection.

**"'ollama' binary not found in PATH"**
The "register in Ollama" button calls the `ollama` command line tool. If
the installer didn't add it to your PATH (occasionally happens on
Windows), restart your session, or reinstall Ollama.

**Download seems stuck or fails partway**
Check available disk space in the models folder, and that
`huggingface_hub` is installed (step 4) for resumable downloads.

**Sizes shown as "[estimate]"**
This means the addon couldn't reach the Hugging Face API during the scan
(no connection, firewall, rate limiting) and fell back to a heuristic
estimate — treat these numbers, and any download attempted from them, as
approximate.

**3D: the runtime install or a download looks stuck**
Use the **Cancel** button; downloads resume where they stopped. Older versions
of this addon ran the official installer *with* its interactive
`Proceed? [Y/n]` prompt (it blocks forever without a console on Windows) and
read its output line by line, while curl's progress bar only emits carriage
returns — so the UI looked frozen. Both are fixed.

**3D: "unreachable" right after Start**
The server answers `/health` before the pipeline is fully loaded; wait for the
*"Serveur prêt"* status. If the URL field still shows `8787` from an older
version, set it to `http://127.0.0.1:8080` (the real default).

**3D: "refused the request (HTTP 500)"**
The message includes the server's own error (typically a missing weight file).
Re-run *Download TRELLIS.2 weights* and check the log in `<3D folder>/logs/`.

---

## 9. Uninstalling

**Edit > Preferences > Add-ons**, find "Local LLM Assistant", expand it
and click **Remove**. Downloaded `.gguf` files and models registered in
Ollama are not deleted automatically — remove them manually from the
models folder and via `ollama rm <name>` if needed.

---

## License

Add your license of choice here before publishing (MIT is a common
default for a small utility addon like this one).