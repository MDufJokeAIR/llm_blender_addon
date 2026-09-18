bl_info = {
    "name": "Local LLM Assistant",
    "author": "Joke",
    "version": (0, 4, 0),
    "blender": (4, 1, 0),
    "location": "View3D > Sidebar (N) > Local LLM",
    "description": (
        "Assistant IA local (Qwen, Llama, Gemma, Phi, Mistral, SmolLM, "
        "TinyLlama, GLM, DeepSeek, Kimi...) via Ollama : recommandation de "
        "modele selon la VRAM disponible, recherche libre, navigation par "
        "famille, telechargement depuis Hugging Face, mode assistant simple "
        "ou controle agentique de la scene. Catalogue aussi une famille "
        "Generation 3D (TRELLIS.2, TRELLIS, Hunyuan3D-2, TripoSR, "
        "InstantMesh, Shap-E) a titre de reference, hors du flux Ollama."
    ),
    "category": "3D View",
}

import bpy
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import urllib.request
import urllib.error

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
    "gen3d": {
        "label": "Generation 3D (image/texte -> mesh)",
        "note": (
            "Modeles de GENERATION D'OBJETS 3D (image ou texte -> mesh texture), pas "
            "des LLM de chat : ils ne passent PAS par Ollama et ce panneau ne sait "
            "pas les executer. Listes ici pour la recherche et la reference, avec un "
            "lien direct vers leur page Hugging Face (installation via un "
            "environnement Python dedie - diffusers/ComfyUI selon le modele)."
        ),
        "chat_compatible": False,
        "models": [
            {"name": "TRELLIS.2-4B", "repo_id": "microsoft/TRELLIS.2-4B", "params_b": 4.0, "non_gguf": True,
             "desc": "Microsoft, MIT. Generation image-vers-3D haute fidelite (topologie arbitraire, materiaux PBR, jusqu'a 1536^3), sortie mesh texture. Tres efficace pour de la conception 3D mais gourmand (~24 Go de VRAM recommandes) et teste surtout sous Linux."},
            {"name": "TRELLIS (image-large)", "repo_id": "microsoft/TRELLIS-image-large", "params_b": 1.2, "non_gguf": True,
             "desc": "Microsoft, MIT. Version precedente de TRELLIS, plus legere que TRELLIS.2, toujours une reference solide pour l'image-vers-3D."},
            {"name": "Hunyuan3D-2", "repo_id": "tencent/Hunyuan3D-2", "params_b": 0.0, "non_gguf": True,
             "desc": "Tencent. Genere des meshes textures haute resolution a partir d'image ou de texte, tres reputee en conception d'assets 3D ; pipeline Diffusers/Safetensors, licence communautaire Tencent-Hunyuan."},
            {"name": "TripoSR", "repo_id": "stabilityai/TripoSR", "params_b": 0.0, "non_gguf": True,
             "desc": "Stability AI. Reconstruction 3D a partir d'une seule image, tres rapide et legere : bon point d'entree si le materiel est modeste."},
            {"name": "InstantMesh", "repo_id": "TencentARC/InstantMesh", "params_b": 0.0, "non_gguf": True,
             "desc": "Tencent ARC, Apache 2.0. Genere un mesh 3D a partir d'une image en environ 10 secondes (diffusion multi-vue + reconstruction LRM)."},
            {"name": "Shap-E", "repo_id": "openai/shap-e", "params_b": 0.0, "non_gguf": True,
             "desc": "OpenAI, MIT. Plus ancien et beaucoup plus leger que les modeles ci-dessus (texte/image -> objet 3D), qualite modeste mais utile sur config tres limitee."},
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
        if not FAMILIES[family_key].get("chat_compatible", True):
            continue  # ex. generation 3D : pas des LLM de chat, hors scope de ce scan
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
    is_llm: bpy.props.BoolProperty(
        default=True,
        description="Faux pour un modele de generation 3D (non pilotable via Ollama)",
    )


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

    # --- Recherche libre (toutes familles confondues, LLM + generation 3D) ---
    search_query: bpy.props.StringProperty(
        name="Rechercher",
        description="Filtre tout le catalogue par nom, famille ou mot-cle "
                    "(ex. 'coder', 'TRELLIS', 'leger') - ignore le budget VRAM",
    )
    searching: bpy.props.BoolProperty(default=False)
    search_status: bpy.props.StringProperty(default="")
    search_results: bpy.props.CollectionProperty(type=LLMModelItem)

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

        if entry.get("non_gguf"):
            # Pas un modele GGUF/Ollama : pas de scan reseau de quantizations,
            # juste une entree pointant vers sa page Hugging Face.
            settings.browse_scanning = False
            settings.browse_results.clear()
            item = settings.browse_results.add()
            item.display_name = entry["name"]
            item.repo_id = entry["repo_id"]
            item.desc = entry.get("desc", "")
            item.is_llm = False
            settings.browse_status = (
                "Pas un modele de chat GGUF compatible Ollama : ce panneau ne "
                "sait pas l'executer. Ouvre sa page Hugging Face (icone lien) "
                "pour son propre environnement d'installation."
            )
            return {'FINISHED'}

        thread = threading.Thread(
            target=_browse_worker,
            args=(entry["repo_id"], entry["params_b"], entry["name"], entry.get("desc", "")),
            daemon=True,
        )
        thread.start()
        bpy.app.timers.register(_poll_browse_result, first_interval=0.3)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Recherche libre - filtre tout le catalogue (LLM + generation 3D) par
# mot-cle, sans tenir compte du budget VRAM (recherche nominative, pas
# recommandation).
# ---------------------------------------------------------------------------

_search_lock = threading.Lock()
_search_buffer = {"done": False, "results": [], "error": None}


def _search_worker(query):
    needle = query.strip().lower()
    matches = []
    for family_key, entry in iter_all_models():
        haystack = " ".join([
            entry["name"], entry["repo_id"], FAMILIES[family_key]["label"],
            entry.get("desc", ""),
        ]).lower()
        if needle in haystack:
            matches.append((family_key, entry))

    results = []
    try:
        for family_key, entry in matches:
            if entry.get("non_gguf"):
                results.append({
                    "name": entry["name"], "family": family_key,
                    "repo_id": entry["repo_id"], "filename": "", "quant": "",
                    "size_gb": 0.0, "online": False, "is_llm": False,
                    "desc": entry.get("desc", ""),
                })
                continue

            options = list_quant_options(entry["repo_id"], entry["params_b"])
            best = None
            for quant in QUANT_PREFERENCE:
                best = next((o for o in options if o[1] == quant), None)
                if best:
                    break
            if best is None and options:
                best = options[0]
            if best:
                fname, quant, size_gb, online = best
                results.append({
                    "name": entry["name"], "family": family_key,
                    "repo_id": entry["repo_id"], "filename": fname,
                    "quant": quant, "size_gb": size_gb, "online": online,
                    "is_llm": True, "desc": entry.get("desc", ""),
                })
        with _search_lock:
            _search_buffer.update(done=True, results=results, error=None)
    except Exception as exc:
        with _search_lock:
            _search_buffer.update(done=True, results=[], error=str(exc))


def _poll_search_result():
    with _search_lock:
        done = _search_buffer["done"]
    if not done:
        return 0.3

    with _search_lock:
        results = _search_buffer["results"]
        error = _search_buffer["error"]
        _search_buffer.update(done=False, results=[], error=None)

    for scene in bpy.data.scenes:
        settings = scene.llm_assistant
        settings.searching = False
        settings.search_results.clear()
        if error:
            settings.search_status = f"Erreur : {error}"
            continue
        for r in results:
            item = settings.search_results.add()
            item.display_name = f"[{FAMILIES[r['family']]['label']}] {r['name']}"
            item.repo_id = r["repo_id"]
            item.filename = r["filename"]
            item.quant = r["quant"]
            item.size_gb = r["size_gb"]
            item.online = r["online"]
            item.is_llm = r["is_llm"]
            item.desc = r.get("desc", "")
        settings.search_status = (
            f"{len(results)} resultat(s)" if results
            else "Aucun modele ne correspond a cette recherche"
        )
    return None


class LLM_OT_search_catalog(bpy.types.Operator):
    """Cherche dans tout le catalogue (LLM et generation 3D) par nom,
    famille ou mot-cle, sans tenir compte du budget VRAM"""
    bl_idname = "llm.search_catalog"
    bl_label = "Rechercher"

    def execute(self, context):
        settings = context.scene.llm_assistant
        if not settings.search_query.strip():
            self.report({'WARNING'}, "Tape un mot-cle a rechercher")
            return {'CANCELLED'}

        settings.searching = True
        settings.search_status = "Recherche en cours..."

        thread = threading.Thread(
            target=_search_worker,
            args=(settings.search_query,),
            daemon=True,
        )
        thread.start()
        bpy.app.timers.register(_poll_search_result, first_interval=0.3)
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


def _draw_model_list(box, settings, collection_name):
    coll = getattr(settings, collection_name)
    for i, item in enumerate(coll):
        row = box.row(align=True)
        if item.is_llm:
            label = f"{item.display_name} - {item.quant} (~{item.size_gb} Go)"
            if not item.online:
                label += " [estimation]"
        else:
            label = f"{item.display_name} [generation 3D, hors Ollama]"
        info_op = row.operator(
            LLM_OT_model_info.bl_idname, text=label, icon='INFO', emboss=False,
        )
        info_op.info_text = item.desc or "Pas de description disponible pour ce modele."

        if not item.is_llm:
            # Modele de generation 3D : pas de flux telechargement/Ollama,
            # juste un lien direct vers sa page Hugging Face.
            url_op = row.operator("wm.url_open", text="", icon='URL')
            url_op.url = f"https://huggingface.co/{item.repo_id}"
        elif item.downloaded:
            op = row.operator(LLM_OT_register_ollama.bl_idname, text="", icon='CHECKMARK')
            op.index = i
            op.collection = collection_name
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

        # --- Recherche libre dans tout le catalogue ---
        box = layout.box()
        box.label(text="Rechercher un modele", icon='VIEWZOOM')
        row = box.row(align=True)
        row.prop(settings, "search_query", text="")
        sub = row.row()
        sub.enabled = not settings.searching
        sub.operator(
            LLM_OT_search_catalog.bl_idname,
            text="..." if settings.searching else "OK",
        )
        if settings.search_status:
            box.label(text=settings.search_status)
        _draw_model_list(box, settings, "search_results")

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
    LLM_OT_search_catalog,
    LLM_OT_download_model,
    LLM_OT_register_ollama,
    LLM_OT_model_info,
    LLM_OT_send_chat,
    LLM_OT_execute_code,
    LLM_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.llm_assistant = bpy.props.PointerProperty(type=LLMAssistantSettings)


def unregister():
    del bpy.types.Scene.llm_assistant
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()