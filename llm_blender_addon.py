bl_info = {
    "name": "Local LLM Assistant",
    "author": "Joke",
    "version": (0, 4, 1),
    "blender": (4, 1, 0),
    "location": "View3D > Sidebar (N) > Local LLM",
    "description": (
        "Assistant IA local (Qwen, Llama, Gemma, Phi, Mistral, SmolLM, "
        "TinyLlama, GLM, DeepSeek, Kimi...) via Ollama, plus generation 3D "
        "locale (image/texte -> mesh) via TRELLIS.2/trellis.cpp et "
        "Z-Image/stable-diffusion.cpp : recommandation de modele selon la "
        "VRAM disponible, navigation par famille, telechargement depuis "
        "Hugging Face, mode assistant simple ou controle agentique de la scene."
    ),
    "category": "3D View",
}

import atexit
import bpy
import collections
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Backend d'inference local, compatible API Ollama.
OLLAMA_URL = "http://localhost:11434"

# Dossier ou les .gguf telecharges sont stockes avant d'etre enregistres
# dans Ollama (modifiable dans le panel).
DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), "llm_blender_models"
)

# ---------------------------------------------------------------------------
# Generation 3D (texte/image -> mesh), pipeline confirme :
#   texte -> image  : stable-diffusion.cpp (Z-Image)  [binaire sd-cli]
#   image -> mesh   : trellis.cpp (TRELLIS.2, Microsoft, MIT)
#
# trellis.cpp (github.com/pwilkin/trellis.cpp) est un runtime C++/ggml
# autonome (equivalent de llama.cpp mais pour TRELLIS.2) : poids GGUF,
# zero Python a l'execution, serveur HTTP resident expose :
#   GET  /health
#   POST /generate   multipart/form-data, champ "image" (+ "seed",
#                     "resolution" 512/1024/1536, "bg_removal"), renvoie
#                     un model/gltf-binary (GLB).
# Poids GGUF recommandes par le projet : ilintar/trellis2-gguf, avec 3
# paliers f16 (~16.5 Go), q8 (~9.5 Go), q4 (~6 Go) geres par sous-dossier.
# ---------------------------------------------------------------------------
TRELLIS_INSTALL_SH = "https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.sh"
TRELLIS_INSTALL_PS1 = "https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.ps1"
TRELLIS_GGUF_REPO = "ilintar/trellis2-gguf"
# Paliers VRAM -> (nom du quant, taille approx en Go, sous-dossier dans le repo).
# Port par defaut de trellis-server : 8080 (verifie dans l'installeur officiel
# et docs/getting-started.md ; l'installeur ecrit aussi host/port dans
# config.json). Le serveur se lance avec : --models DIR --port P [--host H] [--gpu N].
TRELLIS_QUANT_TIERS = [
    ("f16", 16.5, ""),
    ("q8",  9.5,  "q8"),
    ("q4",  6.0,  "q4"),
]
DEFAULT_TRELLIS_SERVER_URL = "http://127.0.0.1:8080"
# Les 10 fichiers GGUF que l'installeur officiel telecharge (verifie dans
# install.sh/install.ps1). Les paliers q8/q4 utilisent les memes noms, dans
# les sous-dossiers q8/ et q4/ du depot ; le serveur attend un dossier PLAT.
TRELLIS_WEIGHT_FILES = [
    "birefnet.gguf", "dinov3.gguf", "ss_flow.gguf", "ss_dec.gguf",
    "shape_flow_512.gguf", "shape_flow_1024.gguf", "shape_dec.gguf",
    "tex_flow_512.gguf", "tex_flow_1024.gguf", "tex_dec.gguf",
]
TRELLIS_HF_BASE = f"https://huggingface.co/{TRELLIS_GGUF_REPO}/resolve/main"

# Z-Image (texte -> image) via sd-cli (stable-diffusion.cpp, leejet). Pas
# d'installeur one-liner confirme pour sd-cli lui-meme : l'addon telecharge
# les POIDS (3 composants, via la meme logique HF que le reste du catalogue)
# mais demande le chemin du binaire sd-cli deja installe par l'utilisateur
# (voir github.com/leejet/stable-diffusion.cpp pour le build/les releases).
ZIMAGE_DIFFUSION_REPO = "leejet/Z-Image-Turbo-GGUF"
ZIMAGE_TEXT_ENCODER_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
ZIMAGE_VAE_REPO = "black-forest-labs/FLUX.1-schnell"
ZIMAGE_VAE_FILE = "ae.safetensors"
# Quants recommandes par palier VRAM pour le diffusion model / l'encodeur texte.
ZIMAGE_QUANT_BY_TIER = {
    "q4": ("Q4_0", "Q4_K_M"),
    "q8": ("Q5_K_M", "Q6_K"),
    "f16": ("Q8_0", "Q8_0"),
}

DEFAULT_3D_DIR = os.path.join(os.path.expanduser("~"), "llm_blender_3d")
DEFAULT_3D_OUTPUT_DIR = os.path.join(DEFAULT_3D_DIR, "output")

# Duree mesuree par le projet (image -> GLB, res 1024, modele charge) : sert
# uniquement a afficher une fourchette indicative, ce n'est pas une
# progression en temps reel (l'API ne l'expose pas).
TRELLIS_ETA_RANGES = {
    "gpu_dedie": "environ 3 a 7 minutes (GPU dedie, type RTX)",
    "igpu": "environ 6 a 13 minutes (iGPU / APU)",
    "apple": "environ 9 minutes (Apple Silicon, Metal)",
}

# ---------------------------------------------------------------------------
# Catalogue des modeles, groupe par famille pour les menus deroulants.
#
# "params_b" est le nombre de parametres nominal (utilise pour l'estimation
# VRAM et le tri qualite), pas une valeur exacte a l'octet pres. Les tailles
# et noms de fichiers REELS sont recuperes en ligne au moment du scan
# (cf. list_repo_gguf_files) ; ce catalogue ne fournit que le repo_id de
# depart. Beaucoup de repos ici sont des quantizations communautaires
# (bartowski, TheBloke...) plutot que les repos officiels des editeurs,
# afin d'eviter les repos "gated" qui demandent d'accepter une licence sur
# Hugging Face avant de pouvoir telecharger quoi que ce soit.
#
# A tenir a jour : de nouvelles familles/versions sortent regulierement, et
# un repo_id errone se degrade proprement (l'entree tombe en mode
# [estimation] plutot que de planter, cf. list_repo_gguf_files).
# ---------------------------------------------------------------------------

FAMILIES = {
    "qwen": {
        "label": "Qwen (Alibaba)",
        "note": "Gamme la plus complete du catalogue (0.6B a 32B, MoE, variantes Coder), licence Apache 2.0.",
        "models": [
            {"name": "Qwen3 0.6B",           "repo_id": "Qwen/Qwen3-0.6B-GGUF",                       "params_b": 0.6,
             "desc": "Tres petit modele generaliste multilingue (licence Apache 2.0). Convient a des taches simples et rapides, capacites limitees sur le raisonnement complexe."},
            {"name": "Qwen3 1.7B",           "repo_id": "Qwen/Qwen3-1.7B-GGUF",                       "params_b": 1.7,
             "desc": "Generaliste multilingue, Apache 2.0. Bon compromis pour un chat d'aide basique sur une petite config."},
            {"name": "Qwen3 4B",             "repo_id": "Qwen/Qwen3-4B-GGUF",                         "params_b": 4.0,
             "desc": "Generaliste multilingue, Apache 2.0. Nettement plus solide en raisonnement que les tailles 0.6-1.7B."},
            {"name": "Qwen3 8B",             "repo_id": "Qwen/Qwen3-8B-GGUF",                         "params_b": 8.0,
             "desc": "Generaliste multilingue, Apache 2.0. Bon equilibre qualite/vitesse sur une carte grand public (8-12 Go)."},
            {"name": "Qwen3 14B",            "repo_id": "Qwen/Qwen3-14B-GGUF",                        "params_b": 14.0,
             "desc": "Generaliste multilingue, Apache 2.0. Meilleur raisonnement, demande davantage de VRAM."},
            {"name": "Qwen3 32B",            "repo_id": "Qwen/Qwen3-32B-GGUF",                        "params_b": 32.0,
             "desc": "Generaliste multilingue haut de gamme de la famille Qwen3, Apache 2.0. Demande une configuration consequente."},
            {"name": "Qwen3 30B-A3B (MoE)",  "repo_id": "Qwen/Qwen3-30B-A3B-GGUF",                    "params_b": 30.0,
             "desc": "Mixture-of-Experts : 30 Md de parametres au total mais seulement ~3 Md actifs par requete (inference plus rapide qu'un dense de meme taille), Apache 2.0. Le fichier a telecharger reste base sur les 30 Md totaux."},
            {"name": "Qwen2.5 Coder 7B",     "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",        "params_b": 7.0,  "coder": True,
             "desc": "Specialise generation/comprehension de code, Apache 2.0. Bon choix pour le mode controle agentique (generation de scripts bpy)."},
            {"name": "Qwen2.5 Coder 14B",    "repo_id": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",       "params_b": 14.0, "coder": True,
             "desc": "Specialise code, plus capable que la version 7B, Apache 2.0. Pour le mode controle agentique."},
            {"name": "Qwen2.5 Coder 32B",    "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",       "params_b": 32.0, "coder": True,
             "desc": "Le plus capable de la gamme Coder Qwen2.5, Apache 2.0. Pour le mode controle agentique, demande une grosse config."},
        ],
    },
    "llama": {
        "label": "Llama (Meta)",
        "note": "Quantizations communautaires (bartowski) : pas besoin d'accepter la licence Meta sur Hugging Face.",
        "models": [
            {"name": "Llama 3.2 1B Instruct",   "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",        "params_b": 1.0,
             "desc": "Meta, tres leger, oriente usage embarque/edge. Licence communautaire Llama."},
            {"name": "Llama 3.2 3B Instruct",   "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",        "params_b": 3.0,
             "desc": "Meta, generaliste leger, meilleur que le 1B en raisonnement. Licence communautaire Llama."},
            {"name": "Llama 3.1 8B Instruct",   "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",   "params_b": 8.0,
             "desc": "Modele generaliste de reference chez Meta a cette taille, tres largement utilise. Licence communautaire Llama."},
            {"name": "Llama 3.3 70B Instruct",  "repo_id": "bartowski/Llama-3.3-70B-Instruct-GGUF",       "params_b": 70.0,
             "desc": "Grand modele Meta, qualite proche des 70B precedents mais optimise. Demande une configuration tres consequente (24-32 Go+ meme quantise). Licence communautaire Llama."},
        ],
    },
    "gemma": {
        "label": "Gemma (Google)",
        "note": "Modeles Google, generalement tres efficaces par rapport a leur taille.",
        "models": [
            {"name": "Gemma 3 270M",  "repo_id": "bartowski/google_gemma-3-270m-it-GGUF", "params_b": 0.27,
             "desc": "Le plus petit modele Google Gemma 3, pense surtout pour du fine-tuning ou de l'embarque tres contraint. Capacites generalistes tres limitees."},
            {"name": "Gemma 3 1B",    "repo_id": "bartowski/google_gemma-3-1b-it-GGUF",   "params_b": 1.0,
             "desc": "Google, leger, correct pour des taches simples sur petite config."},
            {"name": "Gemma 3 4B",    "repo_id": "bartowski/google_gemma-3-4b-it-GGUF",   "params_b": 4.0,
             "desc": "Google, generaliste, supporte aussi l'entree image dans sa version d'origine (le GGUF text-only ici pour le chat)."},
            {"name": "Gemma 3 12B",   "repo_id": "bartowski/google_gemma-3-12b-it-GGUF",  "params_b": 12.0,
             "desc": "Google, generaliste plus capable, demande davantage de VRAM."},
            {"name": "Gemma 3 27B",   "repo_id": "bartowski/google_gemma-3-27b-it-GGUF",  "params_b": 27.0,
             "desc": "Le plus gros de la gamme Gemma 3, generaliste haut de gamme cote Google."},
        ],
    },
    "phi": {
        "label": "Phi (Microsoft)",
        "note": "Modeles Microsoft, licence MIT, reputes pour leur raisonnement/maths a taille reduite.",
        "models": [
            {"name": "Phi-4 Mini (3.8B)", "repo_id": "bartowski/microsoft_Phi-4-mini-instruct-GGUF", "params_b": 3.8,
             "desc": "Microsoft, licence MIT, reputee solide en maths/raisonnement pour sa taille malgre un format compact."},
            {"name": "Phi-4 (14B)",       "repo_id": "bartowski/microsoft_phi-4-GGUF",               "params_b": 14.0,
             "desc": "Microsoft, licence MIT, bon niveau de raisonnement/maths pour sa taille."},
        ],
    },
    "mistral": {
        "label": "Mistral / Ministral",
        "note": "Modeles Mistral AI (France), Apache 2.0 pour Mistral 7B.",
        "models": [
            {"name": "Ministral 8B Instruct",       "repo_id": "bartowski/Ministral-8B-Instruct-2410-GGUF", "params_b": 8.0,
             "desc": "Mistral AI, modele compact oriente usage local/edge, bon support multilingue europeen."},
            {"name": "Mistral 7B Instruct v0.3",    "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",   "params_b": 7.0,
             "desc": "Mistral AI, generaliste tres largement utilise depuis sa sortie, Apache 2.0."},
        ],
    },
    "smollm": {
        "label": "SmolLM (Hugging Face)",
        "note": "Modeles entierement ouverts de Hugging Face (donnees d'entrainement incluses), tres compacts.",
        "models": [
            {"name": "SmolLM2 135M Instruct", "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",   "params_b": 0.135,
             "desc": "Le plus petit modele de Hugging Face, entierement ouvert (donnees d'entrainement incluses). Capacites tres limitees, surtout utile pour tester le pipeline."},
            {"name": "SmolLM2 360M Instruct", "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",   "params_b": 0.36,
             "desc": "Hugging Face, entierement ouvert. Legerement plus capable que le 135M, reste tres limite."},
            {"name": "SmolLM2 1.7B Instruct", "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",   "params_b": 1.7,
             "desc": "Hugging Face, entierement ouvert. Le plus capable de la serie SmolLM2, correct pour des taches simples."},
            {"name": "SmolLM3 3B",            "repo_id": "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF",    "params_b": 3.0,
             "desc": "Hugging Face, generation suivante de SmolLM, generaliste multilingue compact."},
        ],
    },
    "tinyllama": {
        "label": "TinyLlama",
        "note": "Modele communautaire classique, tres leger, capacites limitees.",
        "models": [
            {"name": "TinyLlama 1.1B Chat", "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "params_b": 1.1,
             "desc": "Modele communautaire tres connu et tres leger, capacites limitees. Surtout utile pour tester rapidement une chaine d'inference."},
        ],
    },
    "glm": {
        "label": "GLM (Zhipu)",
        "note": "Modeles Zhipu AI, reconnus pour le function calling / appel d'outils.",
        "models": [
            {"name": "GLM-4 9B Chat", "repo_id": "bartowski/THUDM_glm-4-9b-chat-GGUF", "params_b": 9.0,
             "desc": "Zhipu AI, generaliste reconnu pour son bon support du function calling / appel d'outils."},
        ],
    },
    "deepseek": {
        "label": "DeepSeek (distill R1)",
        "note": "Distillations de DeepSeek-R1 sur des backbones Qwen/Llama, pas des modeles DeepSeek natifs (ceux-ci font plusieurs centaines de Go).",
        "models": [
            {"name": "DeepSeek-R1-Distill-Qwen 1.5B",  "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",  "params_b": 1.5,
             "desc": "Distillation du raisonnement pas-a-pas de DeepSeek-R1 sur un petit backbone Qwen. Meilleur en logique/maths qu'un generaliste de meme taille, mais plus lent (raisonne en plusieurs etapes avant de repondre)."},
            {"name": "DeepSeek-R1-Distill-Llama 8B",   "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",  "params_b": 8.0,
             "desc": "Meme principe que la version Qwen 1.5B mais sur un backbone Llama 8B : raisonnement pas-a-pas plus solide, reponses plus lentes."},
        ],
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "note": "MoE d'environ 1000 Md de parametres au total : ne tient dans aucun budget de ce panneau (1-32 Go), meme tres quantise. Liste a titre informatif.",
        "models": [
            {"name": "Kimi K2 Instruct", "repo_id": "moonshotai/Kimi-K2-Instruct", "params_b": 1000.0,
             "desc": "Immense modele MoE de Moonshot AI (~1000 Md de parametres au total). Hors de portee d'un PC grand public meme en quantization agressive (des centaines de Go necessaires) ; liste ici uniquement a titre informatif."},
        ],
    },
}


def iter_all_models():
    """Genere (family_key, entry) pour chaque modele du catalogue."""
    for family_key, family in FAMILIES.items():
        for entry in family["models"]:
            yield family_key, entry


# Octets par parametre approximatifs selon la quantization GGUF, calibres
# sur des tailles de fichiers publiees (Qwen3-8B/14B officiels). Utilise
# uniquement en repli si l'API Hugging Face est injoignable.
QUANT_BYTES_PER_PARAM = {
    "Q2_K": 0.35,
    "IQ3_XXS": 0.40, "IQ3_XS": 0.42, "Q3_K_S": 0.45, "Q3_K_M": 0.49, "Q3_K_L": 0.52,
    "Q4_K_S": 0.56, "Q4_K_M": 0.60, "Q4_1": 0.62, "Q4_0": 0.58,
    "Q5_K_S": 0.68, "Q5_K_M": 0.70,
    "Q6_K": 0.82,
    "Q8_0": 1.06,
    "F16": 2.00, "BF16": 2.00, "F32": 4.00,
}
# Ordre de preference (meilleure qualite -> plus compacte) pour choisir le
# meilleur quant qui tient dans le budget VRAM.
QUANT_PREFERENCE = [
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_XS", "IQ3_XXS", "Q2_K",
]

CONTEXT_OVERHEAD_GB = 1.0  # marge pour KV-cache + overhead runtime

_QUANT_RE = re.compile(
    r"(Q[2-8](?:_K)?(?:_[SML])?|Q4_[01]|IQ[1-4]_\w+|F16|BF16|F32)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dependances (huggingface_hub) - optionnelles, installables depuis le panel
# ---------------------------------------------------------------------------

def _hf_available():
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


class LLM_OT_install_deps(bpy.types.Operator):
    """Installe huggingface_hub dans le Python de Blender (telechargement
    plus robuste, avec reprise) - facultatif, le scan fonctionne sans"""
    bl_idname = "llm.install_deps"
    bl_label = "Installer huggingface_hub"

    def execute(self, context):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "huggingface_hub",
            ])
        except Exception as exc:
            self.report({'ERROR'}, f"Echec de l'installation : {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, "huggingface_hub installe.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Estimation VRAM / recommandation de modele
# ---------------------------------------------------------------------------

def estimate_size_gb(params_b, quant):
    bpp = QUANT_BYTES_PER_PARAM.get(quant, 0.6)
    return round(params_b * bpp, 3)


def list_repo_gguf_files(repo_id):
    """Retourne [(filename, quant, size_gb), ...] pour un repo HF donne, en
    interrogeant l'API publique Hugging Face directement (endpoint 'tree').
    Ne depend PAS de huggingface_hub, et donne les noms de fichiers REELS
    du repo (contrairement a une estimation qui devine le nom du fichier,
    et peut donc 404 au telechargement). Renvoie None si l'API est
    injoignable (pas de connexion, timeout, repo introuvable...)."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    results = []
    for entry in entries:
        fname = entry.get("path", "")
        if not fname.lower().endswith(".gguf"):
            continue
        match = _QUANT_RE.search(fname)
        quant = match.group(1).upper() if match else "UNKNOWN"
        # Les .gguf sont suivis via Git LFS : la taille reelle est dans
        # lfs.size (le champ "size" au premier niveau n'est que celle du
        # pointeur LFS, quelques centaines d'octets, le cas echeant).
        lfs_info = entry.get("lfs") or {}
        raw_size = lfs_info.get("size") or entry.get("size") or 0
        size_gb = round(raw_size / (1024 ** 3), 3)
        if size_gb <= 0:
            continue
        results.append((fname, quant, size_gb))
    return results


def list_repo_tree(repo_id, path=""):
    """Comme list_repo_gguf_files, mais renvoie TOUS les fichiers (pas
    seulement les .gguf) d'un repo HF, optionnellement sous un sous-dossier
    (utile pour les depots organises en sous-dossiers par quant, ex.
    ilintar/trellis2-gguf avec des dossiers q4/ et q8/). Renvoie
    [(chemin_relatif, taille_gb), ...] ou None si l'API est injoignable."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    if path:
        url += f"/{path.strip('/')}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    results = []
    for entry in entries:
        if entry.get("type") not in (None, "file"):
            continue
        fpath = entry.get("path", "")
        if not fpath:
            continue
        lfs_info = entry.get("lfs") or {}
        raw_size = lfs_info.get("size") or entry.get("size") or 0
        size_gb = round(raw_size / (1024 ** 3), 3)
        results.append((fpath, size_gb))
    return results


def list_quant_options(repo_id, params_b):
    """Renvoie [(filename, quant, size_gb, online), ...] pour un repo
    donne, triee de la meilleure qualite a la plus compacte. Utilise les
    vraies tailles Hugging Face si disponibles, sinon une estimation pour
    chaque quant standard (marquee online=False, nom de fichier devine)."""
    online_files = list_repo_gguf_files(repo_id)
    options = []
    if online_files:
        by_quant = {}
        for fname, quant, size_gb in online_files:
            # En cas de doublon (rare), garde le plus petit fichier pour ce quant.
            if quant not in by_quant or size_gb < by_quant[quant][1]:
                by_quant[quant] = (fname, size_gb)
        for quant in QUANT_PREFERENCE:
            if quant in by_quant:
                fname, size_gb = by_quant[quant]
                options.append((fname, quant, size_gb, True))
    else:
        base = repo_id.split("/")[-1]
        for quant in QUANT_PREFERENCE:
            size_gb = estimate_size_gb(params_b, quant)
            fname = f"{base}-{quant.lower()}.gguf"
            options.append((fname, quant, size_gb, False))
    return options


def recommend_models(vram_budget_gb, prefer_coder=False):
    """Parcourt tout le catalogue (toutes familles) et renvoie une liste
    triee de recommandations qui tiennent dans le budget VRAM (avec marge).
    Chaque item : {name, family, repo_id, filename, quant, size_gb,
    params_b, online}."""
    budget = max(0.3, vram_budget_gb - CONTEXT_OVERHEAD_GB)
    recommendations = []

    for family_key, entry in iter_all_models():
        if prefer_coder and not entry.get("coder"):
            continue

        # Pre-filtre : si meme le quant le plus agressif ne tient pas (avec
        # une marge de securite), inutile d'interroger le reseau pour ce
        # modele -> le scan reste rapide meme avec un catalogue large.
        smallest_possible = estimate_size_gb(entry["params_b"], "Q2_K")
        if smallest_possible > budget * 1.3:
            continue

        options = list_quant_options(entry["repo_id"], entry["params_b"])
        best = next((o for o in options if o[2] <= budget), None)
        if best:
            fname, quant, size_gb, online = best
            recommendations.append({
                "name": entry["name"],
                "family": family_key,
                "repo_id": entry["repo_id"],
                "filename": fname,
                "quant": quant,
                "size_gb": size_gb,
                "params_b": entry["params_b"],
                "online": online,
                "desc": entry.get("desc", ""),
            })

    # Meilleur = le plus gros nombre de parametres qui tient dans le budget.
    recommendations.sort(key=lambda r: r["params_b"], reverse=True)
    return recommendations


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class LLMModelItem(bpy.types.PropertyGroup):
    display_name: bpy.props.StringProperty()
    repo_id: bpy.props.StringProperty()
    filename: bpy.props.StringProperty()
    quant: bpy.props.StringProperty()
    size_gb: bpy.props.FloatProperty()
    params_b: bpy.props.FloatProperty()
    online: bpy.props.BoolProperty()
    downloaded: bpy.props.BoolProperty(default=False)
    local_path: bpy.props.StringProperty()
    desc: bpy.props.StringProperty()


# Caches module-level pour les items d'EnumProperty dynamiques : Blender
# exige de garder une reference vivante sur les chaines des items, sinon
# elles peuvent etre liberees par le garbage collector et planter l'UI.
_family_enum_cache = []
_variant_enum_cache = []


def _family_enum_items(self, context):
    global _family_enum_cache
    _family_enum_cache = [
        (key, fam["label"], fam.get("note", "")) for key, fam in FAMILIES.items()
    ]
    return _family_enum_cache


def _variant_enum_items(self, context):
    global _variant_enum_cache
    family = FAMILIES.get(self.browse_family)
    if not family:
        _variant_enum_cache = [("NONE", "-", "")]
        return _variant_enum_cache
    _variant_enum_cache = [
        (str(i), m["name"], f"{m.get('desc', '')} (repo : {m['repo_id']})")
        for i, m in enumerate(family["models"])
    ]
    return _variant_enum_cache


class LLMAssistantSettings(bpy.types.PropertyGroup):
    vram_budget_gb: bpy.props.IntProperty(
        name="VRAM allouee (Go)",
        description="Quantite de VRAM que tu acceptes de dedier au modele",
        default=8, min=1, max=32,
    )
    mode_simple: bpy.props.BoolProperty(
        name="Assistant simple",
        description="Chat d'aide / generation de code affiche, sans execution automatique",
        default=True,
    )
    mode_agentic: bpy.props.BoolProperty(
        name="Controle agentique",
        description="L'IA peut proposer du code bpy que tu executes en un clic pour agir sur la scene",
        default=False,
    )
    models_dir: bpy.props.StringProperty(
        name="Dossier modeles",
        subtype='DIR_PATH',
        default=DEFAULT_MODELS_DIR,
    )

    # --- Recommandation automatique (toutes familles, selon VRAM) ---
    scanning: bpy.props.BoolProperty(default=False)
    scan_status: bpy.props.StringProperty(default="")
    recommended_models: bpy.props.CollectionProperty(type=LLMModelItem)

    # --- Navigation manuelle par famille / modele ---
    browse_family: bpy.props.EnumProperty(name="Famille", items=_family_enum_items)
    browse_variant: bpy.props.EnumProperty(name="Modele", items=_variant_enum_items)
    browse_scanning: bpy.props.BoolProperty(default=False)
    browse_status: bpy.props.StringProperty(default="")
    browse_results: bpy.props.CollectionProperty(type=LLMModelItem)

    download_status: bpy.props.StringProperty(default="")
    downloading: bpy.props.BoolProperty(default=False)

    active_ollama_model: bpy.props.StringProperty(
        name="Modele actif (Ollama)",
        description="Nom du modele tel qu'enregistre dans Ollama (ollama list)",
        default="llama3.2:3b",
    )
    chat_input: bpy.props.StringProperty(name="Message")
    chat_history: bpy.props.StringProperty(default="")
    chat_busy: bpy.props.BoolProperty(default=False)
    last_code_block: bpy.props.StringProperty(default="")

    # --- Generation 3D (texte/image -> mesh) ---
    gen3d_vram_gb: bpy.props.IntProperty(
        name="VRAM pour la generation 3D (Go)",
        description="Determine le palier de quantization TRELLIS.2 (q4/q8/f16) telecharge",
        default=16, min=1, max=32,
    )
    gen3d_models_dir: bpy.props.StringProperty(
        name="Dossier poids 3D", subtype='DIR_PATH', default=DEFAULT_3D_DIR,
    )
    gen3d_output_dir: bpy.props.StringProperty(
        name="Dossier des GLB generes", subtype='DIR_PATH', default=DEFAULT_3D_OUTPUT_DIR,
    )
    trellis_server_url: bpy.props.StringProperty(
        name="URL serveur 3D",
        description=(
            "Adresse de trellis-server une fois installe et lance. Le port "
            "par defaut n'est pas garanti : verifie la sortie de l'installeur "
            "ou les reglages de Trellis Studio si la connexion echoue"
        ),
        default=DEFAULT_TRELLIS_SERVER_URL,
    )
    trellis_server_bin: bpy.props.StringProperty(
        name="Binaire trellis-server",
        description="Rempli automatiquement apres l'installation du runtime (modifiable)",
        subtype='FILE_PATH', default="",
    )
    trellis_models_dir: bpy.props.StringProperty(
        name="Dossier des poids charges",
        description="Dossier passe a trellis-server (--models) ; rempli apres le telechargement des poids",
        subtype='DIR_PATH', default="",
    )
    trellis_weights_progress: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    zimage_progress: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    trellis_install_status: bpy.props.StringProperty(default="")
    trellis_installing: bpy.props.BoolProperty(default=False)
    trellis_health_status: bpy.props.StringProperty(default="")
    trellis_checking_health: bpy.props.BoolProperty(default=False)
    trellis_weights_status: bpy.props.StringProperty(default="")
    trellis_weights_downloading: bpy.props.BoolProperty(default=False)

    gen3d_mode: bpy.props.EnumProperty(
        name="Source",
        items=[
            ('IMAGE', "Depuis une image", "Mesh genere a partir d'une image existante"),
            ('TEXT', "Depuis du texte", "Image generee par Z-Image puis convertie en mesh"),
        ],
        default='IMAGE',
    )
    gen3d_image_path: bpy.props.StringProperty(
        name="Image source", subtype='FILE_PATH', default="",
    )
    gen3d_prompt: bpy.props.StringProperty(
        name="Prompt", description="Description de l'image a generer avant conversion en mesh",
    )
    sdcli_path: bpy.props.StringProperty(
        name="Binaire sd-cli",
        description="Chemin vers sd-cli (stable-diffusion.cpp), a installer separement",
        subtype='FILE_PATH', default="",
    )
    zimage_status: bpy.props.StringProperty(default="")
    zimage_downloading: bpy.props.BoolProperty(default=False)

    gen3d_status: bpy.props.StringProperty(default="")
    gen3d_busy: bpy.props.BoolProperty(default=False)
    gen3d_start_time: bpy.props.FloatProperty(default=0.0)
    gen3d_progress: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    gen3d_last_glb: bpy.props.StringProperty(default="")


# ---------------------------------------------------------------------------
# Scan operator (recommandation globale par VRAM) - thread + timer
# ---------------------------------------------------------------------------

_scan_lock = threading.Lock()
_scan_result_buffer = {"done": False, "models": [], "error": None}


def _scan_worker(vram_budget, prefer_coder):
    try:
        models = recommend_models(vram_budget, prefer_coder)
        with _scan_lock:
            _scan_result_buffer.update(done=True, models=models, error=None)
    except Exception as exc:
        with _scan_lock:
            _scan_result_buffer.update(done=True, models=[], error=str(exc))


def _poll_scan_result():
    with _scan_lock:
        done = _scan_result_buffer["done"]
    if not done:
        return 0.3

    with _scan_lock:
        models = _scan_result_buffer["models"]
        error = _scan_result_buffer["error"]
        _scan_result_buffer.update(done=False, models=[], error=None)

    for scene in bpy.data.scenes:
        settings = scene.llm_assistant
        settings.scanning = False
        settings.recommended_models.clear()
        if error:
            settings.scan_status = f"Erreur : {error}"
            continue
        for m in models:
            item = settings.recommended_models.add()
            item.display_name = f"[{FAMILIES[m['family']]['label']}] {m['name']}"
            item.repo_id = m["repo_id"]
            item.filename = m["filename"]
            item.quant = m["quant"]
            item.size_gb = m["size_gb"]
            item.params_b = m["params_b"]
            item.online = m["online"]
            item.desc = m.get("desc", "")
        settings.scan_status = (
            f"{len(models)} modele(s) trouve(s)" if models
            else "Aucun modele ne tient dans ce budget (essaie d'augmenter le curseur, "
                 "ou regarde les tres petits modeles via 'Parcourir par famille')"
        )
    return None


class LLM_OT_scan_models(bpy.types.Operator):
    """Cherche, dans TOUT le catalogue, les modeles qui tiennent dans le
    budget VRAM choisi"""
    bl_idname = "llm.scan_models"
    bl_label = "Scanner les modeles disponibles"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if not _hf_available():
            self.report(
                {'INFO'},
                "huggingface_hub absent : le scan reste precis (API Hugging "
                "Face interrogee directement), mais le telechargement n'aura "
                "pas de reprise en cas de coupure.",
            )
        settings.scanning = True
        settings.scan_status = "Scan en cours (peut prendre quelques dizaines de secondes)..."

        prefer_coder = settings.mode_agentic and not settings.mode_simple

        thread = threading.Thread(
            target=_scan_worker,
            args=(settings.vram_budget_gb, prefer_coder),
            daemon=True,
        )
        thread.start()
        bpy.app.timers.register(_poll_scan_result, first_interval=0.3)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Browse operator (un seul modele choisi via les menus deroulants)
# ---------------------------------------------------------------------------

_browse_lock = threading.Lock()
_browse_buffer = {"done": False, "options": [], "error": None, "name": "", "repo_id": "", "desc": ""}


def _browse_worker(repo_id, params_b, name, desc):
    try:
        options = list_quant_options(repo_id, params_b)
        with _browse_lock:
            _browse_buffer.update(done=True, options=options, error=None, name=name, repo_id=repo_id, desc=desc)
    except Exception as exc:
        with _browse_lock:
            _browse_buffer.update(done=True, options=[], error=str(exc), name=name, repo_id=repo_id, desc=desc)


def _poll_browse_result():
    with _browse_lock:
        done = _browse_buffer["done"]
    if not done:
        return 0.3

    with _browse_lock:
        options = _browse_buffer["options"]
        error = _browse_buffer["error"]
        name = _browse_buffer["name"]
        repo_id = _browse_buffer["repo_id"]
        desc = _browse_buffer["desc"]
        _browse_buffer.update(done=False, options=[], error=None)

    for scene in bpy.data.scenes:
        settings = scene.llm_assistant
        settings.browse_scanning = False
        settings.browse_results.clear()
        if error:
            settings.browse_status = f"Erreur : {error}"
            continue
        for fname, quant, size_gb, online in options:
            item = settings.browse_results.add()
            item.display_name = name
            item.repo_id = repo_id
            item.filename = fname
            item.quant = quant
            item.size_gb = size_gb
            item.online = online
            item.desc = desc
        settings.browse_status = (
            f"{len(options)} quantization(s) trouvee(s)" if options
            else "Aucune info trouvee pour ce modele"
        )
    return None


class LLM_OT_browse_scan(bpy.types.Operator):
    """Recupere les tailles de fichiers disponibles pour le modele choisi
    dans les menus Famille / Modele ci-dessus"""
    bl_idname = "llm.browse_scan"
    bl_label = "Voir les tailles disponibles"

    def execute(self, context):
        settings = context.scene.llm_assistant
        family = FAMILIES.get(settings.browse_family)
        if not family:
            self.report({'ERROR'}, "Choisis une famille")
            return {'CANCELLED'}
        try:
            entry = family["models"][int(settings.browse_variant)]
        except (ValueError, IndexError):
            self.report({'ERROR'}, "Choisis un modele")
            return {'CANCELLED'}

        settings.browse_scanning = True
        settings.browse_status = "Recherche en cours..."

        thread = threading.Thread(
            target=_browse_worker,
            args=(entry["repo_id"], entry["params_b"], entry["name"], entry.get("desc", "")),
            daemon=True,
        )
        thread.start()
        bpy.app.timers.register(_poll_browse_result, first_interval=0.3)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Download / register operators - generalises pour agir sur n'importe
# quelle collection (recommended_models ou browse_results)
# ---------------------------------------------------------------------------

_download_lock = threading.Lock()
_download_buffer = {
    "done": False, "path": None, "error": None, "index": -1,
    "collection": "recommended_models",
}


def _download_worker(repo_id, filename, target_dir, index, collection):
    try:
        if _hf_available():
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=target_dir,
            )
        else:
            # Repli : telechargement HTTP direct (fonctionne pour les repos
            # publics, sans les fonctionnalites de resume de huggingface_hub).
            os.makedirs(target_dir, exist_ok=True)
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            path = os.path.join(target_dir, filename)
            urllib.request.urlretrieve(url, path)
        with _download_lock:
            _download_buffer.update(done=True, path=path, error=None, index=index, collection=collection)
    except Exception as exc:
        with _download_lock:
            _download_buffer.update(done=True, path=None, error=str(exc), index=index, collection=collection)


def _poll_download_result():
    with _download_lock:
        done = _download_buffer["done"]
    if not done:
        return 0.3

    with _download_lock:
        path = _download_buffer["path"]
        error = _download_buffer["error"]
        index = _download_buffer["index"]
        collection = _download_buffer["collection"]
        _download_buffer.update(done=False, path=None, error=None, index=-1)

    for scene in bpy.data.scenes:
        settings = scene.llm_assistant
        settings.downloading = False
        coll = getattr(settings, collection, None)
        if coll is None:
            continue
        if error:
            settings.download_status = f"Erreur telechargement : {error}"
            continue
        settings.download_status = f"Telecharge : {path}"
        if 0 <= index < len(coll):
            item = coll[index]
            item.downloaded = True
            item.local_path = path
    return None


class LLM_OT_download_model(bpy.types.Operator):
    """Telecharge le fichier GGUF selectionne depuis Hugging Face"""
    bl_idname = "llm.download_model"
    bl_label = "Telecharger"

    index: bpy.props.IntProperty()
    collection: bpy.props.StringProperty(default="recommended_models")

    def execute(self, context):
        settings = context.scene.llm_assistant
        coll = getattr(settings, self.collection, None)
        if coll is None or self.index < 0 or self.index >= len(coll):
            self.report({'ERROR'}, "Selection invalide")
            return {'CANCELLED'}

        item = coll[self.index]
        if not item.online:
            self.report(
                {'WARNING'},
                "Taille/nom de fichier estimes (API Hugging Face injoignable "
                "pendant le scan) : le telechargement peut echouer en 404. "
                "Relance le scan avec une connexion active pour un resultat fiable.",
            )
        settings.downloading = True
        settings.download_status = f"Telechargement de {item.filename}..."

        thread = threading.Thread(
            target=_download_worker,
            args=(item.repo_id, item.filename, settings.models_dir, self.index, self.collection),
            daemon=True,
        )
        thread.start()
        bpy.app.timers.register(_poll_download_result, first_interval=0.3)
        return {'FINISHED'}


class LLM_OT_register_ollama(bpy.types.Operator):
    """Enregistre le .gguf telecharge comme modele Ollama utilisable"""
    bl_idname = "llm.register_ollama"
    bl_label = "Enregistrer dans Ollama"

    index: bpy.props.IntProperty()
    collection: bpy.props.StringProperty(default="recommended_models")

    def execute(self, context):
        settings = context.scene.llm_assistant
        coll = getattr(settings, self.collection, None)
        if coll is None or self.index < 0 or self.index >= len(coll):
            self.report({'ERROR'}, "Selection invalide")
            return {'CANCELLED'}

        item = coll[self.index]
        if not item.downloaded or not item.local_path:
            self.report({'ERROR'}, "Telecharge d'abord le modele")
            return {'CANCELLED'}

        model_name = re.sub(r"[^a-z0-9._-]+", "-", item.display_name.lower()).strip("-")
        modelfile_path = item.local_path + ".Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(f'FROM "{item.local_path}"\n')

        try:
            subprocess.check_call(["ollama", "create", model_name, "-f", modelfile_path])
        except FileNotFoundError:
            self.report({'ERROR'}, "Binaire 'ollama' introuvable dans le PATH")
            return {'CANCELLED'}
        except subprocess.CalledProcessError as exc:
            self.report({'ERROR'}, f"Echec 'ollama create' : {exc}")
            return {'CANCELLED'}

        settings.active_ollama_model = model_name
        self.report({'INFO'}, f"Modele '{model_name}' pret dans Ollama")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Generation 3D : installation de trellis.cpp, telechargement des poids
# (TRELLIS.2 + Z-Image), et generation du mesh (image ou texte -> GLB).
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _redraw_all():
    """Force le rafraichissement de l'UI (les proprietes modifiees depuis un
    timer ne redessinent pas toujours le panneau toutes seules)."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:
        pass


def _run_logged(cmd, on_line, proc_holder=None, cwd=None, env=None):
    """Lance une commande SANS stdin (un prompt interactif ne peut donc pas
    la bloquer indefiniment) et appelle on_line(texte) pour chaque ligne de
    sortie. La sortie est lue par blocs et decoupee sur \\r ET \\n : les
    barres de progression (curl, sd-cli...) n'emettent que des \\r, ce qui
    figeait l'affichage avec une lecture ligne par ligne.
    Renvoie (succes, fin_de_sortie)."""
    kwargs = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, cwd=cwd, env=env, **kwargs,
        )
    except Exception as exc:
        return False, str(exc)
    if proc_holder is not None:
        proc_holder["proc"] = proc

    tail = []

    def flush(text):
        for part in re.split(r"[\r\n]+", text):
            part = _ANSI_RE.sub("", part).strip()
            if part:
                on_line(part)
                tail.append(part)

    stream = proc.stdout
    pending = ""
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        pending += chunk.decode("utf-8", "replace")
        parts = re.split(r"[\r\n]", pending)
        pending = parts.pop()  # fragment incomplet, complete au prochain bloc
        flush("\n".join(parts))
    flush(pending)
    return proc.wait() == 0, "\n".join(tail[-40:])


class _Job:
    """Etat partage thread de travail <-> timer d'UI (remplace les buffers
    + verrous ecrits a la main)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.state = {"done": False, "status": "", "error": "", "result": ""}

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def get(self):
        with self.lock:
            return dict(self.state)


def _make_poller(job, apply_fn, interval=0.4):
    """Fabrique un callback pour bpy.app.timers : applique l'etat du job aux
    reglages de chaque scene, redessine l'UI, et s'arrete une fois termine."""
    def _poll():
        st = job.get()
        for scene in bpy.data.scenes:
            apply_fn(scene.llm_assistant, st)
        _redraw_all()
        if st["done"]:
            job.reset()
            return None
        return interval
    return _poll


class _Cancelled(Exception):
    pass


_cancel_event = threading.Event()
_installer_proc = {"proc": None}


class _ProgressTracker:
    """Cumule les octets telecharges et en deduit debit et temps restant
    (debit moyen sur les ~15 dernieres secondes)."""

    def __init__(self, total_bytes):
        self.total = total_bytes
        self.done = 0
        self._lock = threading.Lock()
        self._samples = collections.deque(maxlen=256)
        self._samples.append((time.time(), 0))

    def add(self, n):
        with self._lock:
            self.done += n
            self._samples.append((time.time(), self.done))

    def snapshot(self):
        """Renvoie (octets_faits, debit_o_s, secondes_restantes|None, fraction|-1)."""
        now = time.time()
        with self._lock:
            done = self.done
            t_old, d_old = self._samples[0]
            for t, d in self._samples:
                if now - t <= 15:
                    t_old, d_old = t, d
                    break
        dt = now - t_old
        speed = (done - d_old) / dt if dt > 0.5 else 0.0
        remaining = (self.total - done) / speed if (speed > 0 and self.total > done) else None
        fraction = done / self.total if self.total > 0 else -1.0
        return done, speed, remaining, fraction


def _fmt_size(n):
    return f"{n / 1024 ** 3:.2f} Go" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} Mo"


def _fmt_eta(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def _progress_text(tracker, label):
    done, speed, remaining, _ = tracker.snapshot()
    mo_s = f"{speed / 1024 ** 2:.1f} Mo/s"
    if tracker.total > 0:
        return (f"{label} - {_fmt_size(done)} / {_fmt_size(tracker.total)} - "
                f"{mo_s} - reste ~{_fmt_eta(remaining)}")
    return f"{label} - {_fmt_size(done)} - {mo_s}"


def _download_stream(url, dest, on_bytes=None, retries=3, chunk=1024 * 1024):
    """Telechargement HTTP en flux, avec reprise (Range) et progression
    octet par octet. Ecrit dans dest + '.part' puis renomme : un fichier
    portant son nom final est donc toujours complet. Annulable via
    _cancel_event. Pas de dependance a huggingface_hub."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    reported = 0  # octets deja signales a on_bytes pour CE fichier
    last_exc = None
    for _attempt in range(retries):
        if _cancel_event.is_set():
            raise _Cancelled()
        resume_from = os.path.getsize(part) if os.path.isfile(part) else 0
        headers = {"User-Agent": "llm-blender-addon"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and resume_from:  # le .part etait deja complet
                os.replace(part, dest)
                return
            if exc.code in (401, 403):
                raise RuntimeError(f"Acces refuse (HTTP {exc.code}) : {url} (depot prive ou soumis a licence ?)")
            if exc.code == 404:
                raise RuntimeError(f"Fichier introuvable (404) : {url}")
            last_exc = exc
            time.sleep(2)
            continue
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            time.sleep(2)
            continue
        try:
            with resp:
                if resume_from and getattr(resp, "status", 200) != 206:
                    resume_from = 0  # le serveur ignore Range : on repart de zero
                    reported = 0
                if resume_from and on_bytes and resume_from > reported:
                    on_bytes(resume_from - reported)
                    reported = resume_from
                with open(part, "ab" if resume_from else "wb") as f:
                    while True:
                        if _cancel_event.is_set():
                            raise _Cancelled()
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        reported += len(block)
                        if on_bytes:
                            on_bytes(len(block))
            os.replace(part, dest)
            return
        except _Cancelled:
            raise
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            time.sleep(2)
    raise RuntimeError(f"Telechargement echoue apres {retries} tentatives : {last_exc}")


def _pick_tier(vram_gb):
    """Palier TRELLIS.2 (nom, taille, sous-dossier) pour un budget VRAM."""
    for tier in TRELLIS_QUANT_TIERS:
        if vram_gb >= tier[1] - 2:
            return tier
    return TRELLIS_QUANT_TIERS[-1]


# --- Config / chemins trellis.cpp -------------------------------------------

def _trellis_config_path():
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "trellis-studio", "config.json")


def _find_trellis_server_bin():
    """Cherche trellis-server : d'abord via le config.json que l'installeur
    ecrit (cle serverBin), puis a l'emplacement d'installation par defaut."""
    candidates = []
    try:
        with open(_trellis_config_path(), encoding="utf-8-sig") as f:
            candidates.append(json.load(f).get("serverBin", ""))
    except Exception:
        pass
    if sys.platform.startswith("win"):
        dest = os.path.join(os.environ.get("LOCALAPPDATA", ""), "trellis-studio")
        exe = "trellis-server.exe"
    else:
        dest = os.path.join(
            os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
            "trellis-studio",
        )
        exe = "trellis-server"
    candidates.append(os.path.join(dest, "runtime", exe))
    return next((c for c in candidates if c and os.path.isfile(c)), "")


# --- 1. Installation du runtime ---------------------------------------------

_install_job = _Job()


def _trellis_install_worker():
    try:
        if sys.platform == "darwin":
            raise RuntimeError(
                "L'installeur officiel couvre Linux et Windows. Sur macOS : "
                "recupere ou compile trellis.cpp (voir son depot GitHub), puis "
                "renseigne 'Binaire trellis-server' a la main."
            )
        windows = sys.platform.startswith("win")
        _install_job.update(status="Telechargement du script d'installation...")
        req = urllib.request.Request(
            TRELLIS_INSTALL_PS1 if windows else TRELLIS_INSTALL_SH,
            headers={"User-Agent": "llm-blender-addon"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        script = os.path.join(
            tempfile.mkdtemp(prefix="trellis_install_"),
            "install.ps1" if windows else "install.sh",
        )
        if windows and not data.startswith(b"\xef\xbb\xbf"):
            data = b"\xef\xbb\xbf" + data  # PowerShell 5 lit le UTF-8 sans BOM comme de l'ANSI
        with open(script, "wb") as f:
            f.write(data)

        # Options NON INTERACTIVES, indispensables : sans -Yes / -y le script
        # affiche "Proceed? [Y/n]" et attend une reponse que personne ne peut
        # donner depuis Blender -> blocage indefini. --skip-models : les poids
        # sont telecharges par l'addon (palier VRAM + progression/ETA) ;
        # --skip-app : l'appli Trellis Studio n'est pas necessaire ici.
        if windows:
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-File", script, "-Yes", "-SkipModels", "-SkipApp"]
        else:
            cmd = ["bash", script, "-y", "--skip-models", "--skip-app"]
        ok, tail = _run_logged(
            cmd, lambda line: _install_job.update(status=line),
            proc_holder=_installer_proc,
        )
        if _cancel_event.is_set():
            raise _Cancelled()
        # Le code de sortie seul n'est pas fiable : avec --skip-app, la derniere
        # ligne d'install.sh (`[ -f ...AppImage ] && info ...`) renvoie 1 meme
        # quand tout s'est bien passe (constate en test). Critere retenu :
        # echec si le serveur reste introuvable, ou si le script a signale
        # lui-meme une ligne "error: ...".
        server_bin = _find_trellis_server_bin()
        script_reported_error = any(
            l.lower().startswith("error") for l in tail.splitlines()
        )
        if not server_bin or (not ok and script_reported_error):
            raise RuntimeError(f"L'installeur a echoue : {tail[-400:]}")
        _install_job.update(done=True, status="Runtime installe.", result=server_bin)
    except _Cancelled:
        _install_job.update(done=True, error="Annule.")
    except Exception as exc:
        _install_job.update(done=True, error=str(exc))


def _apply_install(settings, st):
    if not st["done"]:
        settings.trellis_install_status = st["status"] or "Installation en cours..."
        return
    settings.trellis_installing = False
    if st["error"]:
        settings.trellis_install_status = f"Echec : {st['error']}"
    else:
        settings.trellis_install_status = "Runtime installe (poids a telecharger a l'etape suivante)."
        settings.trellis_server_bin = st["result"]


class LLM_OT_install_trellis(bpy.types.Operator):
    """Installe le runtime trellis.cpp (serveur uniquement, quelques dizaines
    de Mo) via l'installeur officiel du projet, en mode non interactif. Les
    poids sont telecharges separement, selon ton budget VRAM."""
    bl_idname = "llm.install_trellis"
    bl_label = "Installer le runtime trellis.cpp"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if settings.trellis_installing:
            return {'CANCELLED'}
        _cancel_event.clear()
        _install_job.reset()
        settings.trellis_installing = True
        settings.trellis_install_status = "Lancement..."
        threading.Thread(target=_trellis_install_worker, daemon=True).start()
        bpy.app.timers.register(_make_poller(_install_job, _apply_install), first_interval=0.4)
        return {'FINISHED'}


class LLM_OT_cancel_3d_tasks(bpy.types.Operator):
    """Annule l'installation / les telechargements 3D en cours (les fichiers
    deja telecharges sont conserves : la reprise est automatique)"""
    bl_idname = "llm.cancel_3d_tasks"
    bl_label = "Annuler"

    def execute(self, context):
        _cancel_event.set()
        proc = _installer_proc.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return {'FINISHED'}


# --- 2. Poids TRELLIS.2 -----------------------------------------------------

_weights_job = _Job()


def _trellis_weights_worker(subfolder, target_dir):
    try:
        _weights_job.update(status="Lecture du depot Hugging Face...")
        sizes = {}
        listing = list_repo_tree(TRELLIS_GGUF_REPO, path=subfolder)
        for path, size_gb in (listing or []):
            sizes[os.path.basename(path)] = int(size_gb * 1024 ** 3)
        known = all(sizes.get(n, 0) > 0 for n in TRELLIS_WEIGHT_FILES)
        tracker = _ProgressTracker(sum(sizes[n] for n in TRELLIS_WEIGHT_FILES) if known else 0)
        os.makedirs(target_dir, exist_ok=True)
        prefix = f"{subfolder}/" if subfolder else ""
        count = len(TRELLIS_WEIGHT_FILES)

        for i, name in enumerate(TRELLIS_WEIGHT_FILES, 1):
            _weights_job.update(tracker=tracker, label=f"{i}/{count} {name}")
            dest = os.path.join(target_dir, name)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                tracker.add(os.path.getsize(dest))  # deja telecharge (relance)
                continue
            _download_stream(f"{TRELLIS_HF_BASE}/{prefix}{name}", dest, on_bytes=tracker.add)

        _weights_job.update(
            done=True, result=target_dir,
            status=f"Poids TRELLIS.2 prets ({count} fichiers) : {target_dir}",
        )
    except _Cancelled:
        _weights_job.update(
            done=True, result="",
            status="Annule. Les fichiers deja telecharges sont conserves (reprise automatique).",
        )
    except Exception as exc:
        _weights_job.update(done=True, error=str(exc))


def _apply_weights(settings, st):
    tracker = st.get("tracker")
    if not st["done"]:
        if tracker is not None:
            settings.trellis_weights_status = _progress_text(tracker, st.get("label", ""))
            settings.trellis_weights_progress = max(tracker.snapshot()[3], 0.0)
        else:
            settings.trellis_weights_status = st["status"]
        return
    settings.trellis_weights_downloading = False
    if st["error"]:
        settings.trellis_weights_status = f"Erreur : {st['error']}"
    else:
        settings.trellis_weights_status = st["status"]
        if st["result"]:
            settings.trellis_weights_progress = 1.0
            settings.trellis_models_dir = st["result"]


class LLM_OT_download_trellis_weights(bpy.types.Operator):
    """Telecharge les 10 fichiers de poids TRELLIS.2 pour le palier choisi
    par le curseur VRAM (reprise automatique, progression et temps restant)"""
    bl_idname = "llm.download_trellis_weights"
    bl_label = "Telecharger les poids TRELLIS.2"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if settings.trellis_weights_downloading:
            return {'CANCELLED'}
        quant, size_gb, subfolder = _pick_tier(settings.gen3d_vram_gb)
        target_dir = os.path.join(bpy.path.abspath(settings.gen3d_models_dir), f"trellis-{quant}")
        _cancel_event.clear()
        _weights_job.reset()
        settings.trellis_weights_downloading = True
        settings.trellis_weights_progress = 0.0
        settings.trellis_weights_status = f"Palier {quant} (~{size_gb} Go) - preparation..."
        threading.Thread(
            target=_trellis_weights_worker, args=(subfolder, target_dir), daemon=True,
        ).start()
        bpy.app.timers.register(_make_poller(_weights_job, _apply_weights), first_interval=0.4)
        return {'FINISHED'}


# --- 3. Serveur : demarrage / arret / verification ---------------------------

_health_job = _Job()


def _health_worker(url):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", "ignore").strip().lower()
        _health_job.update(done=True, result="ok" if body.startswith("ok") else body[:60])
    except Exception as exc:
        _health_job.update(done=True, error=str(exc))


def _apply_health(settings, st):
    if not st["done"]:
        return
    settings.trellis_checking_health = False
    if st["error"]:
        settings.trellis_health_status = (
            f"Injoignable ({st['error']}). Le serveur est-il demarre ? Port a verifier."
        )
    elif st["result"] == "ok":
        settings.trellis_health_status = "Serveur joignable (/health -> ok)"
    else:
        settings.trellis_health_status = f"Reponse inattendue : {st['result']}"


class LLM_OT_check_trellis_health(bpy.types.Operator):
    """Verifie que trellis-server repond sur l'URL configuree"""
    bl_idname = "llm.check_trellis_health"
    bl_label = "Verifier la connexion"

    def execute(self, context):
        settings = context.scene.llm_assistant
        _health_job.reset()
        settings.trellis_checking_health = True
        settings.trellis_health_status = "Verification..."
        threading.Thread(
            target=_health_worker, args=(settings.trellis_server_url,), daemon=True,
        ).start()
        bpy.app.timers.register(_make_poller(_health_job, _apply_health), first_interval=0.3)
        return {'FINISHED'}


_trellis_proc = {"proc": None, "log_path": "", "url": "", "until": 0.0}


def _stop_trellis_proc():
    proc = _trellis_proc.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
    _trellis_proc["proc"] = None


atexit.register(_stop_trellis_proc)


def _tail_file(path, max_chars=400):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            text = f.read().decode("utf-8", "replace")
        text = _ANSI_RE.sub("", text)
        return " | ".join(l.strip() for l in text.splitlines() if l.strip())[-max_chars:]
    except Exception:
        return ""


def _poll_server_ready():
    proc = _trellis_proc.get("proc")
    finished = True
    if proc is None:
        status = "Serveur arrete."
    elif proc.poll() is not None:
        status = (f"Le serveur s'est arrete (code {proc.returncode}). Log : "
                  f"{_trellis_proc['log_path']} - {_tail_file(_trellis_proc['log_path'])}")
    else:
        status = None
        try:
            with urllib.request.urlopen(_trellis_proc["url"].rstrip("/") + "/health", timeout=1) as r:
                if r.read().decode("utf-8", "ignore").strip().lower().startswith("ok"):
                    status = "Serveur pret (/health -> ok)"
        except Exception:
            pass
        if status is None:
            if time.time() > _trellis_proc["until"]:
                status = f"Toujours en chargement apres 3 min - voir le log : {_trellis_proc['log_path']}"
            else:
                status = "Chargement du pipeline TRELLIS.2 sur le GPU..."
                finished = False
    for scene in bpy.data.scenes:
        scene.llm_assistant.trellis_health_status = status
    _redraw_all()
    return None if finished else 2.0


class LLM_OT_start_trellis_server(bpy.types.Operator):
    """Lance trellis-server en arriere-plan avec les poids telecharges (le
    chargement initial sur le GPU peut prendre un moment)"""
    bl_idname = "llm.start_trellis_server"
    bl_label = "Demarrer le serveur 3D"

    def execute(self, context):
        settings = context.scene.llm_assistant
        proc = _trellis_proc.get("proc")
        if proc is not None and proc.poll() is None:
            self.report({'INFO'}, "Le serveur est deja lance")
            return {'CANCELLED'}

        bin_path = bpy.path.abspath(settings.trellis_server_bin) or _find_trellis_server_bin()
        if not bin_path or not os.path.isfile(bin_path):
            self.report({'ERROR'}, "trellis-server introuvable : lance d'abord l'etape 1 (installation du runtime)")
            return {'CANCELLED'}
        models_dir = bpy.path.abspath(settings.trellis_models_dir)
        if not models_dir or not os.path.isdir(models_dir):
            self.report({'ERROR'}, "Dossier de poids introuvable : lance d'abord l'etape 2 (telechargement des poids)")
            return {'CANCELLED'}
        missing = [f for f in TRELLIS_WEIGHT_FILES if not os.path.isfile(os.path.join(models_dir, f))]
        if missing:
            self.report({'ERROR'}, f"Poids incomplets, il manque : {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
            return {'CANCELLED'}

        parsed = urllib.parse.urlparse(settings.trellis_server_url)
        port = parsed.port or 8080
        log_dir = os.path.join(bpy.path.abspath(settings.gen3d_models_dir), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, time.strftime("server_%Y%m%d_%H%M%S.log"))
        bin_dir = os.path.dirname(bin_path)
        env = os.environ.copy()
        kwargs = {}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        try:
            log_f = open(log_path, "ab")
            _trellis_proc["proc"] = subprocess.Popen(
                [bin_path, "--host", "127.0.0.1", "--port", str(port), "--models", models_dir],
                stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=bin_dir, env=env, **kwargs,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Lancement impossible : {exc}")
            return {'CANCELLED'}
        _trellis_proc.update(
            log_path=log_path, url=f"http://127.0.0.1:{port}", until=time.time() + 180,
        )
        settings.trellis_server_url = f"http://127.0.0.1:{port}"
        settings.trellis_health_status = "Demarrage du serveur..."
        bpy.app.timers.register(_poll_server_ready, first_interval=2.0)
        return {'FINISHED'}


class LLM_OT_stop_trellis_server(bpy.types.Operator):
    """Arrete le trellis-server lance depuis cet addon (libere la VRAM)"""
    bl_idname = "llm.stop_trellis_server"
    bl_label = "Arreter le serveur 3D"

    def execute(self, context):
        _stop_trellis_proc()
        context.scene.llm_assistant.trellis_health_status = "Serveur arrete."
        return {'FINISHED'}


class LLM_OT_pick_image(bpy.types.Operator):
    """Ouvre le navigateur de fichiers pour choisir l'image source"""
    bl_idname = "llm.pick_image"
    bl_label = "Choisir une image"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.png;*.jpg;*.jpeg;*.webp", options={'HIDDEN'})

    def execute(self, context):
        context.scene.llm_assistant.gen3d_image_path = self.filepath
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


_zimage_job = _Job()


def _pick_quant_option(options, wanted_quant):
    """Option (fichier, quant, taille, online) au quant voulu ; a defaut la
    mieux classee, plutot que d'echouer."""
    for opt in options:
        if opt[1] == wanted_quant:
            return opt
    return options[0] if options else None


def _zimage_worker(diff_quant, txt_quant, target_dir):
    try:
        _zimage_job.update(status="Lecture des depots Hugging Face...")
        plan = []  # (repo, fichier, role, taille_octets)
        for repo, quant, role, params in (
            (ZIMAGE_DIFFUSION_REPO, diff_quant, "diffusion", 6.0),
            (ZIMAGE_TEXT_ENCODER_REPO, txt_quant, "text_encoder", 4.0),
        ):
            opt = _pick_quant_option(list_quant_options(repo, params), quant)
            if not opt or not opt[3]:
                raise RuntimeError(f"Depot {repo} injoignable (connexion ?) : nom de fichier indeterminable")
            plan.append((repo, opt[0], role, int(opt[2] * 1024 ** 3)))
        vae_gb = next(
            (sz for p, sz in (list_repo_tree(ZIMAGE_VAE_REPO) or []) if p == ZIMAGE_VAE_FILE), 0,
        )
        plan.append((ZIMAGE_VAE_REPO, ZIMAGE_VAE_FILE, "vae", int(vae_gb * 1024 ** 3)))

        known = all(item[3] > 0 for item in plan)
        tracker = _ProgressTracker(sum(item[3] for item in plan) if known else 0)
        os.makedirs(target_dir, exist_ok=True)
        manifest = {}
        for i, (repo, fname, role, _size) in enumerate(plan, 1):
            _zimage_job.update(tracker=tracker, label=f"{i}/{len(plan)} {role}")
            dest = os.path.join(target_dir, os.path.basename(fname))
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                tracker.add(os.path.getsize(dest))
            else:
                url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(fname)}"
                _download_stream(url, dest, on_bytes=tracker.add)
            manifest[role] = dest

        # Chemin exact de chaque composant enregistre ici plutot que redevine
        # a la generation (noms de fichiers trop varies pour un motif fiable).
        with open(os.path.join(target_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        _zimage_job.update(done=True, result=target_dir,
                           status=f"Composants Z-Image prets : {target_dir}")
    except _Cancelled:
        _zimage_job.update(done=True, result="",
                           status="Annule. Les fichiers deja telecharges sont conserves.")
    except Exception as exc:
        _zimage_job.update(done=True, error=str(exc))


def _apply_zimage(settings, st):
    tracker = st.get("tracker")
    if not st["done"]:
        if tracker is not None:
            settings.zimage_status = _progress_text(tracker, st.get("label", ""))
            settings.zimage_progress = max(tracker.snapshot()[3], 0.0)
        else:
            settings.zimage_status = st["status"]
        return
    settings.zimage_downloading = False
    settings.zimage_status = f"Erreur : {st['error']}" if st["error"] else st["status"]
    if not st["error"] and st["result"]:
        settings.zimage_progress = 1.0


class LLM_OT_download_zimage(bpy.types.Operator):
    """Telecharge les 3 composants Z-Image (diffusion, encodeur texte,
    VAE) necessaires a sd-cli pour le texte -> image, au quant adapte au
    curseur VRAM (reprise automatique, progression et temps restant)"""
    bl_idname = "llm.download_zimage"
    bl_label = "Telecharger les poids Z-Image"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if settings.zimage_downloading:
            return {'CANCELLED'}
        quant = _pick_tier(settings.gen3d_vram_gb)[0]
        diff_quant, txt_quant = ZIMAGE_QUANT_BY_TIER.get(quant, ZIMAGE_QUANT_BY_TIER["q4"])
        target_dir = os.path.join(bpy.path.abspath(settings.gen3d_models_dir), "zimage")
        _cancel_event.clear()
        _zimage_job.reset()
        settings.zimage_downloading = True
        settings.zimage_progress = 0.0
        settings.zimage_status = "Preparation..."
        threading.Thread(
            target=_zimage_worker, args=(diff_quant, txt_quant, target_dir), daemon=True,
        ).start()
        bpy.app.timers.register(_make_poller(_zimage_job, _apply_zimage), first_interval=0.4)
        return {'FINISHED'}


def _multipart_post(url, fields, file_field, file_path, timeout=1800):
    """POST multipart/form-data minimal (stdlib uniquement) vers
    trellis-server. Renvoie (bytes_reponse, content_type)."""
    boundary = uuid.uuid4().hex
    body = io.BytesIO()

    def write(s):
        body.write(s.encode("utf-8") if isinstance(s, str) else s)

    for key, value in fields.items():
        write(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n')
    filename = os.path.basename(file_path)
    write(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    )
    with open(file_path, "rb") as f:
        body.write(f.read())
    write(f"\r\n--{boundary}--\r\n")

    data = body.getvalue()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


_gen3d_lock = threading.Lock()
_gen3d_buffer = {"done": False, "glb_path": "", "error": "", "status": ""}


def _gen3d_worker(settings_snapshot):
    (mode, prompt, image_path, sdcli_path, server_url,
     output_dir, models_dir) = settings_snapshot
    try:
        if mode == 'TEXT':
            if not sdcli_path or not os.path.isfile(sdcli_path):
                raise RuntimeError("Chemin vers sd-cli invalide (renseigne 'Binaire sd-cli')")
            manifest_path = os.path.join(models_dir, "zimage", "manifest.json")
            if not os.path.isfile(manifest_path):
                raise RuntimeError(
                    "Composants Z-Image introuvables - clique d'abord sur "
                    "'Telecharger les poids Z-Image'"
                )
            with open(manifest_path) as f:
                manifest = json.load(f)
            missing = [k for k in ("diffusion", "text_encoder", "vae") if k not in manifest]
            if missing:
                raise RuntimeError(f"Composants Z-Image incomplets, manque : {', '.join(missing)}")

            with _gen3d_lock:
                _gen3d_buffer["status"] = "Generation de l'image (Z-Image via sd-cli)..."
            tmp_png = os.path.join(tempfile.gettempdir(), f"llm_gen3d_{uuid.uuid4().hex}.png")
            # Syntaxe confirmee par le projet stable-diffusion.cpp pour Z-Image.
            cmd = [
                sdcli_path,
                "--diffusion-model", manifest["diffusion"],
                "--llm", manifest["text_encoder"],
                "--vae", manifest["vae"],
                "-p", prompt, "-o", tmp_png,
                "--cfg-scale", "1.0", "--diffusion-fa", "--offload-to-cpu",
            ]
            def _on_sd_line(line):
                with _gen3d_lock:
                    _gen3d_buffer["status"] = f"Z-Image : {line}"[:120]

            ok, output = _run_logged(cmd, _on_sd_line)
            if not ok or not os.path.isfile(tmp_png):
                raise RuntimeError(f"Echec sd-cli : {output[-300:]}")
            image_path = tmp_png

        if not image_path or not os.path.isfile(image_path):
            raise RuntimeError("Aucune image source valide")

        with _gen3d_lock:
            _gen3d_buffer["status"] = "Envoi a trellis-server (/generate)..."
        try:
            glb_bytes, _ = _multipart_post(
                server_url.rstrip("/") + "/generate",
                fields={"resolution": "1024"},
                file_field="image", file_path=image_path,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"trellis-server a refuse la requete (HTTP {exc.code}) : {detail}")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"trellis-server injoignable ({exc.reason}) - demarre-le (etape 3) "
                "et verifie l'URL/le port"
            )

        os.makedirs(output_dir, exist_ok=True)
        glb_path = os.path.join(output_dir, f"mesh_{uuid.uuid4().hex[:8]}.glb")
        with open(glb_path, "wb") as f:
            f.write(glb_bytes)

        with _gen3d_lock:
            _gen3d_buffer.update(done=True, glb_path=glb_path, error="")
    except Exception as exc:
        with _gen3d_lock:
            _gen3d_buffer.update(done=True, error=str(exc))


_STAGE_RE = re.compile(r"\[(\d+)/(\d+)\]\s*([^|\r\n]+)")


def _server_stage():
    """Derniere etape [k/n] affichee dans le log de trellis-server (uniquement
    si c'est cet addon qui l'a lance). Renvoie (texte, fraction) ou None."""
    path = _trellis_proc.get("log_path")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8192))
            text = _ANSI_RE.sub("", f.read().decode("utf-8", "replace"))
    except Exception:
        return None
    matches = _STAGE_RE.findall(text)
    if not matches:
        return None
    k, n, label = matches[-1]
    k, n = int(k), int(n)
    return f"Etape {k}/{n} : {label.strip()[:60]}", max(0.0, (k - 1) / n)


def _poll_gen3d():
    with _gen3d_lock:
        done = _gen3d_buffer["done"]
        status = _gen3d_buffer["status"]

    for scene in bpy.data.scenes:
        settings = scene.llm_assistant
        if not done:
            elapsed = int(time.time() - settings.gen3d_start_time)
            stage = _server_stage()
            if stage and status.startswith("Envoi"):
                status, settings.gen3d_progress = stage
            settings.gen3d_status = f"{status} (ecoule : {elapsed}s)"
        else:
            with _gen3d_lock:
                glb_path = _gen3d_buffer["glb_path"]
                error = _gen3d_buffer["error"]
                _gen3d_buffer.update(done=False, glb_path="", error="", status="")
            settings.gen3d_busy = False
            if error:
                settings.gen3d_status = f"Erreur : {error}"
            else:
                settings.gen3d_status = f"Mesh genere : {glb_path}"
                settings.gen3d_last_glb = glb_path
                try:
                    bpy.ops.import_scene.gltf(filepath=glb_path)
                except Exception as exc:
                    settings.gen3d_status += f" (import automatique echoue : {exc} - importe-le manuellement via File > Import > glTF 2.0)"
    _redraw_all()
    if done:
        return None
    return 1.0


class LLM_OT_generate_mesh(bpy.types.Operator):
    """Lance la generation du mesh 3D (image ou texte -> GLB) et l'importe
    dans la scene. Tourne en arriere-plan ; trellis-server doit deja etre
    lance et joignable (voir 'Verifier la connexion')."""
    bl_idname = "llm.generate_mesh"
    bl_label = "Generer le mesh"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if settings.gen3d_mode == 'IMAGE' and not settings.gen3d_image_path:
            self.report({'ERROR'}, "Choisis d'abord une image source")
            return {'CANCELLED'}
        if settings.gen3d_mode == 'TEXT' and not settings.gen3d_prompt.strip():
            self.report({'ERROR'}, "Ecris d'abord un prompt")
            return {'CANCELLED'}

        settings.gen3d_busy = True
        settings.gen3d_start_time = time.time()
        settings.gen3d_status = "Demarrage..."
        settings.gen3d_progress = 0.0

        snapshot = (
            settings.gen3d_mode, settings.gen3d_prompt, settings.gen3d_image_path,
            settings.sdcli_path, settings.trellis_server_url, settings.gen3d_output_dir,
            settings.gen3d_models_dir,
        )
        thread = threading.Thread(target=_gen3d_worker, args=(snapshot,), daemon=True)
        thread.start()
        bpy.app.timers.register(_poll_gen3d, first_interval=0.5)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Chat (assistant simple / controle agentique)
# ---------------------------------------------------------------------------

_chat_lock = threading.Lock()
_chat_buffer = {"done": False, "reply": None, "error": None}


def _chat_worker(model, user_message, agentic):
    system = (
        "Tu es un assistant integre a Blender. Tu peux proposer du code "
        "Python bpy dans un bloc ```python``` quand c'est pertinent."
        if agentic else
        "Tu es un assistant d'aide a l'utilisation de Blender. Reponds de "
        "maniere concise, avec des extraits de code si utile."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        reply = data.get("message", {}).get("content", "")
        with _chat_lock:
            _chat_buffer.update(done=True, reply=reply, error=None)
    except urllib.error.URLError as exc:
        with _chat_lock:
            _chat_buffer.update(done=True, reply=None, error=f"Ollama injoignable ({exc})")
    except Exception as exc:
        with _chat_lock:
            _chat_buffer.update(done=True, reply=None, error=str(exc))


def _poll_chat_result():
    with _chat_lock:
        done = _chat_buffer["done"]
    if not done:
        return 0.3

    with _chat_lock:
        reply = _chat_buffer["reply"]
        error = _chat_buffer["error"]
        _chat_buffer.update(done=False, reply=None, error=None)

    for scene in bpy.data.scenes:
        settings = scene.llm_assistant
        settings.chat_busy = False
        if error:
            settings.chat_history += f"\n[Erreur] {error}\n"
            continue
        settings.chat_history += f"\nAssistant: {reply}\n"
        match = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
        settings.last_code_block = match.group(1).strip() if match else ""
    return None


class LLM_OT_send_chat(bpy.types.Operator):
    """Envoie le message au modele local via Ollama"""
    bl_idname = "llm.send_chat"
    bl_label = "Envoyer"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if not settings.chat_input.strip():
            return {'CANCELLED'}

        settings.chat_history += f"\nToi: {settings.chat_input}\n"
        settings.chat_busy = True
        message = settings.chat_input
        settings.chat_input = ""

        thread = threading.Thread(
            target=_chat_worker,
            args=(settings.active_ollama_model, message, settings.mode_agentic),
            daemon=True,
        )
        thread.start()
        bpy.app.timers.register(_poll_chat_result, first_interval=0.2)
        return {'FINISHED'}


class LLM_OT_execute_code(bpy.types.Operator):
    """Execute le dernier bloc de code Python propose par l'assistant.
    A n'utiliser qu'en mode controle agentique et apres relecture du code :
    ce code s'execute avec les memes droits que Blender lui-meme."""
    bl_idname = "llm.execute_code"
    bl_label = "Executer le code propose"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.llm_assistant
        code = settings.last_code_block
        if not code:
            self.report({'WARNING'}, "Aucun code a executer")
            return {'CANCELLED'}
        try:
            exec(code, {"bpy": bpy})
        except Exception as exc:
            self.report({'ERROR'}, f"Erreur d'execution : {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, "Code execute")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class LLM_OT_model_info(bpy.types.Operator):
    """Bouton d'information : n'a aucun effet au clic, sert uniquement a
    afficher une description au survol (tooltip dynamique via la methode
    description(), pattern standard de l'API Blender pour ce cas)"""
    bl_idname = "llm.model_info"
    bl_label = ""
    bl_options = {'INTERNAL'}

    info_text: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        return properties.info_text or "Pas de description disponible pour ce modele."

    def execute(self, context):
        return {'FINISHED'}


def _draw_progress(layout, factor):
    """Barre de progression native de Blender (repli sur un label si l'API
    UILayout.progress n'existe pas dans cette version)."""
    factor = max(0.0, min(1.0, factor))
    try:
        layout.progress(factor=factor, type='BAR', text=f"{int(factor * 100)} %")
    except Exception:
        layout.label(text=f"Progression : {int(factor * 100)} %")


def _draw_model_list(box, settings, collection_name):
    coll = getattr(settings, collection_name)
    for i, item in enumerate(coll):
        row = box.row(align=True)
        label = f"{item.display_name} - {item.quant} (~{item.size_gb} Go)"
        if not item.online:
            label += " [estimation]"
        info_op = row.operator(
            LLM_OT_model_info.bl_idname, text=label, icon='INFO', emboss=False,
        )
        info_op.info_text = item.desc or "Pas de description disponible pour ce modele."
        if item.downloaded:
            op = row.operator(LLM_OT_register_ollama.bl_idname, text="", icon='CHECKMARK')
        else:
            op = row.operator(LLM_OT_download_model.bl_idname, text="", icon='IMPORT')
        op.index = i
        op.collection = collection_name


class LLM_PT_panel(bpy.types.Panel):
    bl_label = "Local LLM"
    bl_idname = "LLM_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Local LLM"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.llm_assistant

        # --- Choix du mode ---
        box = layout.box()
        box.label(text="Mode", icon='TOOL_SETTINGS')
        row = box.row()
        row.prop(settings, "mode_simple", toggle=True)
        row.prop(settings, "mode_agentic", toggle=True)

        # --- Recommandation automatique selon la VRAM ---
        box = layout.box()
        box.label(text="Recommandation selon ta VRAM", icon='MEMORY')
        box.prop(settings, "vram_budget_gb", slider=True)
        box.prop(settings, "models_dir")

        if not _hf_available():
            box.operator(LLM_OT_install_deps.bl_idname, icon='IMPORT')

        row = box.row()
        row.enabled = not settings.scanning
        row.operator(
            LLM_OT_scan_models.bl_idname,
            text="Scan en cours..." if settings.scanning else "Scanner tout le catalogue",
            icon='VIEWZOOM',
        )
        if settings.scan_status:
            for line in textwrap.wrap(settings.scan_status, 42):
                box.label(text=line)

        _draw_model_list(box, settings, "recommended_models")

        # --- Navigation manuelle par famille ---
        box = layout.box()
        box.label(text="Parcourir par famille", icon='COLLECTION_NEW')
        box.prop(settings, "browse_family")
        if settings.browse_family:
            box.prop(settings, "browse_variant")
            fam = FAMILIES.get(settings.browse_family)
            if fam and fam.get("note"):
                note_lines = textwrap.wrap(fam["note"], 42)
                for idx, line in enumerate(note_lines):
                    box.label(text=line, icon='INFO' if idx == 0 else 'BLANK1')

            row = box.row()
            row.enabled = not settings.browse_scanning
            row.operator(
                LLM_OT_browse_scan.bl_idname,
                text="Recherche en cours..." if settings.browse_scanning else "Voir les tailles disponibles",
            )
            if settings.browse_status:
                box.label(text=settings.browse_status)

            _draw_model_list(box, settings, "browse_results")

        if settings.download_status:
            for line in textwrap.wrap(settings.download_status, 42):
                layout.label(text=line)

        # --- Generation 3D ---
        box = layout.box()
        box.label(text="Generation 3D (image/texte -> mesh)", icon='MESH_MONKEY')
        box.label(text="Pipeline : TRELLIS.2 via trellis.cpp (MIT)")
        busy_3d = (settings.trellis_installing or settings.trellis_weights_downloading
                   or settings.zimage_downloading)
        if busy_3d:
            box.operator(LLM_OT_cancel_3d_tasks.bl_idname, icon='CANCEL')

        sub = box.box()
        sub.label(text="1. Runtime trellis.cpp", icon='NETWORK_DRIVE')
        row = sub.row()
        row.enabled = not settings.trellis_installing
        row.operator(
            LLM_OT_install_trellis.bl_idname,
            text="Installation en cours..." if settings.trellis_installing else "Installer le runtime (leger)",
        )
        if settings.trellis_install_status:
            for line in textwrap.wrap(settings.trellis_install_status, 42):
                sub.label(text=line)
        sub.prop(settings, "trellis_server_bin")

        sub = box.box()
        sub.label(text="2. Poids TRELLIS.2", icon='MOD_MESHDEFORM')
        sub.prop(settings, "gen3d_vram_gb", slider=True)
        tier = _pick_tier(settings.gen3d_vram_gb)
        sub.label(text=f"Palier retenu : {tier[0]} (~{tier[1]} Go a telecharger)")
        sub.prop(settings, "gen3d_models_dir")
        row = sub.row()
        row.enabled = not settings.trellis_weights_downloading
        row.operator(
            LLM_OT_download_trellis_weights.bl_idname,
            text="Telechargement..." if settings.trellis_weights_downloading else "Telecharger les poids TRELLIS.2",
        )
        if settings.trellis_weights_downloading:
            _draw_progress(sub, settings.trellis_weights_progress)
        if settings.trellis_weights_status:
            for line in textwrap.wrap(settings.trellis_weights_status, 42):
                sub.label(text=line)

        sub = box.box()
        sub.label(text="3. Serveur", icon='PLAY')
        sub.prop(settings, "trellis_models_dir")
        sub.prop(settings, "trellis_server_url")
        row = sub.row(align=True)
        row.operator(LLM_OT_start_trellis_server.bl_idname, icon='PLAY')
        row.operator(LLM_OT_stop_trellis_server.bl_idname, text="", icon='PAUSE')
        row = sub.row()
        row.enabled = not settings.trellis_checking_health
        row.operator(LLM_OT_check_trellis_health.bl_idname)
        if settings.trellis_health_status:
            for line in textwrap.wrap(settings.trellis_health_status, 42):
                sub.label(text=line)

        sub = box.box()
        sub.label(text="4. Source", icon='IMAGE_DATA')
        sub.prop(settings, "gen3d_mode", expand=True)
        if settings.gen3d_mode == 'IMAGE':
            row = sub.row(align=True)
            row.prop(settings, "gen3d_image_path", text="")
            row.operator(LLM_OT_pick_image.bl_idname, text="", icon='FILEBROWSER')
        else:
            sub.prop(settings, "gen3d_prompt")
            sub.prop(settings, "sdcli_path")
            row = sub.row()
            row.enabled = not settings.zimage_downloading
            row.operator(
                LLM_OT_download_zimage.bl_idname,
                text="Telechargement..." if settings.zimage_downloading else "Telecharger les poids Z-Image",
            )
            if settings.zimage_downloading:
                _draw_progress(sub, settings.zimage_progress)
            if settings.zimage_status:
                for line in textwrap.wrap(settings.zimage_status, 42):
                    sub.label(text=line)

        sub = box.box()
        sub.label(text="5. Generation", icon='PLAY')
        sub.prop(settings, "gen3d_output_dir")
        row = sub.row()
        row.enabled = not settings.gen3d_busy
        row.operator(
            LLM_OT_generate_mesh.bl_idname,
            text="Generation en cours..." if settings.gen3d_busy else "Generer le mesh",
        )
        if settings.gen3d_busy:
            _draw_progress(sub, settings.gen3d_progress)
            sub.label(text="Estimation (benchmarks publies) :")
            sub.label(text=TRELLIS_ETA_RANGES["gpu_dedie"])
        if settings.gen3d_status:
            for line in textwrap.wrap(settings.gen3d_status, 42):
                sub.label(text=line)

        # --- Chat ---
        box = layout.box()
        box.label(text="Chat", icon='OUTLINER_OB_LIGHT')
        box.prop(settings, "active_ollama_model")
        col = box.column()
        for line in settings.chat_history.strip().split("\n")[-20:]:
            col.label(text=line)
        row = box.row(align=True)
        row.prop(settings, "chat_input", text="")
        sub = row.row()
        sub.enabled = not settings.chat_busy
        sub.operator(LLM_OT_send_chat.bl_idname, text="Envoyer" if not settings.chat_busy else "...")

        if settings.mode_agentic and settings.last_code_block:
            code_box = box.box()
            code_box.label(text="Code propose :", icon='SCRIPT')
            for line in settings.last_code_block.split("\n")[:10]:
                code_box.label(text=line)
            code_box.operator(LLM_OT_execute_code.bl_idname, icon='PLAY')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    LLMModelItem,
    LLMAssistantSettings,
    LLM_OT_install_deps,
    LLM_OT_scan_models,
    LLM_OT_browse_scan,
    LLM_OT_download_model,
    LLM_OT_register_ollama,
    LLM_OT_model_info,
    LLM_OT_install_trellis,
    LLM_OT_cancel_3d_tasks,
    LLM_OT_start_trellis_server,
    LLM_OT_stop_trellis_server,
    LLM_OT_check_trellis_health,
    LLM_OT_download_trellis_weights,
    LLM_OT_pick_image,
    LLM_OT_download_zimage,
    LLM_OT_generate_mesh,
    LLM_OT_send_chat,
    LLM_OT_execute_code,
    LLM_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.llm_assistant = bpy.props.PointerProperty(type=LLMAssistantSettings)


def unregister():
    _stop_trellis_proc()
    del bpy.types.Scene.llm_assistant
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()