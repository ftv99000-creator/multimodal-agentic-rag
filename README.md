# 🔎 RAG multimodal agentique — « Attention Is All You Need »

Pipeline de **RAG multimodal** (texte + images) appliqué à l'article fondateur du Transformer
([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), piloté par un **agent ReAct** construit avec LangGraph.

L'agent peut répondre à des questions sur le contenu de l'article, retrouver les figures correspondantes,
ou identifier à quelle partie de l'article correspond une image fournie par l'utilisateur.

---

## ✨ Fonctionnalités

- **Extraction multimodale du PDF** : texte découpé en chunks, images raster et figures vectorielles rendues en PNG.
- **Embeddings CLIP** (ViT-B-32) : texte et images dans un **espace latent commun** de dimension 512.
- **Base vectorielle ChromaDB** persistante, avec indexation idempotente (hash SHA-256 du PDF).
- **Recherche multimodale** : requête par **texte** ou par **image**, résultats texte et/ou image.
- **`retriever_tool`** : outil LangChain avec schéma d'arguments typé, exposé au LLM.
- **Agent ReAct (LangGraph)** : architecture MIND (LLM) / BODY (exécution des outils).
- **Batterie de tests** : salutation, question factuelle, requête hybride, comparaison multi-étapes, requête par image.

---

## 🏗️ Architecture

```
                 ┌────────────────────────────────────────────┐
  PDF  ───────▶  │ Extraction : chunks de texte + images      │
                 └───────────────┬────────────────────────────┘
                                 ▼
                 ┌────────────────────────────────────────────┐
                 │ CLIP ViT-B-32 (open_clip) → vecteurs L2    │
                 └───────────────┬────────────────────────────┘
                                 ▼
                 ┌────────────────────────────────────────────┐
                 │ ChromaDB (collection « pdf_multimodal »)   │
                 │ métrique cosinus, champ type = text/image  │
                 └───────────────┬────────────────────────────┘
                                 ▼
                          retriever_tool
                                 ▲
              ┌──────────────────┴──────────────────┐
              │      Agent ReAct (LangGraph)        │
              │                                     │
              │   START → MIND ⇄ BODY → END         │
              │          (LLM)  (ToolNode)          │
              └─────────────────────────────────────┘
```

**Boucle ReAct** : `MIND (Reasoning) → BODY (Action) → retour au MIND (Observation) → … → réponse finale`.
L'arête conditionnelle `tools_condition` route vers BODY si le LLM émet un appel d'outil, sinon vers la fin.

---

## 📁 Structure du dépôt

```
multimodal-agentic-rag/
├── rag_multimodal_agentique.ipynb   # Notebook principal (pipeline complet)
├── Attention is all you need.pdf    # Document source
├── README.md
└── rag_multimodal/                  # Généré à l'exécution
    ├── images/                      # Figures extraites du PDF (PNG)
    └── chroma_db/                   # Base vectorielle ChromaDB persistante
```

---

## ⚙️ Prérequis

- **Python 3.10+**
- Un LLM compatible *tool calling*. Dans la version actuelle du notebook, le LLM par défaut est
  **[Ollama](https://ollama.com/)** en local avec le modèle `qwen2.5:7b`.
- (Optionnel) GPU CUDA ou Apple Silicon (MPS) pour accélérer CLIP ; le CPU fonctionne aussi.

### Installation

```bash
# 1. Dépendances Python
pip install pymupdf open_clip_torch torch torchvision chromadb pillow matplotlib numpy \
            langchain-core langgraph langchain-ollama

# 2. Modèle LLM local
ollama pull qwen2.5:7b
```

> Le notebook exécute lui-même la cellule `%pip install` : cette étape manuelle n'est nécessaire que si vous travaillez hors notebook.

---

## 🚀 Utilisation

1. Lancer Jupyter :
   ```bash
   jupyter notebook "rag_multimodal_agentique.ipynb"
   ```
2. Vérifier qu'Ollama tourne (`ollama serve`) et que le modèle `qwen2.5:7b` est téléchargé.
3. Exécuter les cellules dans l'ordre.

Le fichier `Attention is all you need.pdf` est cherché à côté du notebook ; s'il est absent,
il est **téléchargé automatiquement depuis arXiv**.

### Exemple d'appel direct

```python
# Recherche texte
search("scaled dot-product attention", k=3)

# Recherche par image
search(Path("rag_multimodal/images/page03_raster0.png"), k=3)

# Question à l'agent
ask("Quel score BLEU le Transformer (big) obtient-il sur WMT 2014 EN→DE ?")
```

---

## 🧩 Détail du pipeline

### 1. Extraction

| Contenu | Méthode | Paramètres |
|---|---|---|
| **Texte** | PyMuPDF, découpage par page puis en chunks avec chevauchement | `CHUNK_WORDS = 60`, `OVERLAP = 15` |
| **Images raster** | `page.get_images()` + dédoublonnage MD5 | `MIN_IMG_SIZE = 120 px` |
| **Figures vectorielles** | `page.cluster_drawings()` rendues en PNG (150 dpi) | idem |

Les chunks sont volontairement courts : l'encodeur texte de CLIP est limité à **77 tokens** et tronque silencieusement le reste.
Les légendes (`Figure N: …`) détectées sur chaque page sont associées aux images de cette page.

### 2. Embeddings CLIP

- Modèle : **ViT-B-32**, poids `laion2b_s34b_b79k` (via `open_clip`).
- Vecteurs **normalisés L2** → le produit scalaire équivaut à la similarité cosinus.
- Sélection automatique du device : `cuda` → `mps` → `cpu`.

### 3. Indexation ChromaDB

- `doc_id` = SHA-256 du contenu du fichier : pas de double indexation si le PDF est inchangé ; réindexation propre s'il change.
- Une **seule collection** pour texte et images, filtrable via le champ `type` (`text` / `image`).
- Métrique `cosine` (Chroma renvoie une distance = 1 − similarité).

### 4. Recherche multimodale

La requête peut être un `str`, un `Path` ou une `PIL.Image`. En raison du **modality gap** de CLIP
(les scores texte↔texte et texte↔image n'ont pas la même échelle), les **k meilleurs résultats sont récupérés
séparément par modalité**, puis renvoyés ensemble, plutôt que fusionnés dans un classement unique.

### 5. `retriever_tool`

| Argument | Type | Description |
|---|---|---|
| `query` | `str` | Requête en langage naturel (de préférence en anglais) |
| `image_path` | `str` | Chemin d'une image locale servant de requête |
| `modality` | `"all"` \| `"text"` \| `"image"` | Modalité(s) à interroger |
| `k` | `int` (1–8) | Nombre de résultats par modalité |

La sortie est un texte structuré (`[TEXTE | page N | cos=0.xx] …` / `[IMAGE | page N | fichier=…]`) que le LLM peut citer.

### 6. Agent ReAct

Le **system prompt** impose : utiliser l'outil avant toute réponse factuelle, ne jamais inventer de chiffres,
citer les numéros de page, reformuler et réessayer si besoin, répondre sans outil aux salutations,
et répondre dans la langue de l'utilisateur.

---

## 🧪 Tests et grille d'analyse

| Test | Comportement attendu |
|---|---|
| 1. Salutation | 0 appel d'outil (le MIND répond directement) |
| 2. Scores BLEU | ≥ 1 appel `modality='text'` ou `all`, réponse avec numéro de page, aucun chiffre inventé |
| 3. Hybride texte + figure | Appel avec `modality='all'`, réponse citant la figure retrouvée |
| 4. Comparaison encodeur / décodeur | ≥ 2 appels avec des requêtes reformulées (boucle multi-tours) |
| 5. Requête par image | Appel avec `image_path` renseigné |

La fonction `ask()` rejoue la trace complète : 🧠 *Thought* → 🔧 *Action* → 👁️ *Observation* → ✅ *Réponse finale*.

---

## ⚠️ Limites connues

- **CLIP n'est pas un modèle de texte scientifique** : entraîné sur des paires image-légende, il est moins fin
  qu'un modèle d'embedding textuel dédié pour la recherche texte→texte.
- **Chunks courts** (limite de 77 tokens) : le contexte par passage est restreint.
- **Modality gap** : impose un classement séparé par modalité.
- **Le LLM ne « voit » pas les images** : il reçoit seulement leur chemin et leur légende, sauf à utiliser un modèle multimodal.
- Le modèle local `qwen2.5:7b` peut être moins fiable sur le tool calling qu'un LLM hébergé de grande taille.

---

## 🔧 Pistes d'amélioration

- Ajouter un second index avec un **modèle d'embedding textuel dédié** (ex. BGE, E5) pour le texte.
- Ajouter un **reranker** sur les passages récupérés.
- Passer à un **LLM multimodal** pour que l'agent puisse réellement analyser les figures.
- Brancher d'autres fournisseurs LLM (Anthropic, OpenAI) dans `get_llm()` via la variable `LLM_PROVIDER`.
- Étendre à **plusieurs documents** (le champ `doc_id` est déjà prévu pour cela).

---

## 🛠️ Stack technique

`PyMuPDF` · `open_clip` · `PyTorch` · `ChromaDB` · `LangChain Core` · `LangGraph` · `Ollama` · `Matplotlib` · `Pillow`

---

## 📚 Référence

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I.
**Attention Is All You Need**. *NeurIPS 2017*. <https://arxiv.org/abs/1706.03762>
