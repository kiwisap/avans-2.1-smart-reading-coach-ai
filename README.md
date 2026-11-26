# Smart Reading Coach AI

Het AI prototype (LU2) van **Vrij lezen op maat**. Op basis van het leesprofiel en de leesgeschiedenis van een MBO of NT2
student geeft het model **3 tot 5 leesadviezen met uitleg** waarom ze passen. Het model draait als losse FastAPI service,
zodat de Fastify backend (`smart-reading-coach-api`) het later kan aanroepen in plaats van de huidige vaste regels.

Inhoud: [Workshops](#workshops-en-notebooks) · [Structuur](#mappenstructuur) · [Installeren](#installeren) ·
[Gebruiken](#gebruiken) · [Het model](#het-model) · [Resultaten](#resultaten) · [API](#api) ·
[LU2 checklist](#lu2-checklist) · [Beperkingen](#beperkingen-en-volgende-stappen)

## Workshops en notebooks

Elke workshop uit de LU2 repository ([AvansICT/SE-2.1-2.2-LU2](https://github.com/AvansICT/SE-2.1-2.2-LU2)) en de relevante
workshops van vorig jaar ([AvansICT/ICT1.4-AI-System-Integration-Datascience](https://github.com/AvansICT/ICT1.4-AI-System-Integration-Datascience))
zijn toegepast op de dataset van de opdrachtgever. Doorloop de notebooks op volgorde.

| Notebook | Workshop | Wat gebeurt er |
| --- | --- | --- |
| `01_data_analyse_en_cleaning` | LU2 01, vorig jaar 01, 02, 03 | Missende waarden, duplicaten, datatypes, uitschieters, univariate en bivariate analyse, chi-kwadraat, betrouwbaarheidsinterval, cleaning |
| `02_dimensiereductie_en_clustering` | LU2 02, 03 en K-means uit 04 | PCA, t-SNE en UMAP op tekst en kenmerken, parameters variëren, studentsegmenten met K-means en silhouette score |
| `03_nlp_tekstverwerking` | LU2 05 | Nederlandse NLP pipeline: normaliseren, tokenizen, stopwoorden, stemming, Bag of Words, TF-IDF, synoniemen voor interesses |
| `04_content_based_recommender` | LU2 06 | Cosine similarity, "lijkt op", zoeken op tekst, profiel als zoekvraag |
| `05_collaborative_filtering` | Opdracht LU2 (KNN, matrix factorisatie) | Student x materiaal matrix, item KNN, SVD met fold in, eerlijke tijdsplitsing |
| `06_hybride_model_training_en_evaluatie` | LU2 04, vorig jaar 04, 05, 06 | Logistische regressie, beslisboom, random forest, gradient boosting en neuraal netwerk vergelijken, GridSearchCV, feature importance, kalibratie, vergelijking met baselines, eerlijkheid per groep |
| `07_eindmodel_aanbevelingen_en_uitleg` | Opdracht LU2 | Eindmodel trainen en opslaan, adviezen met uitleg, cold start, docentweergave |

De workshops over object detectie (YOLO) van vorig jaar zijn niet gebruikt: er zitten geen afbeeldingen in deze opdracht.

## Mappenstructuur

```
smart-reading-coach-ai/
  data/
    raw/                  dataset van de opdrachtgever (niet in git)
    processed/            opgeschoonde csv bestanden (gegenereerd)
  models/                 getraind model en metrics.json (gegenereerd)
  notebooks/              de workshops toegepast op de dataset
  reports/figures/        grafieken uit de notebooks
  src/leescoach/          het Python package met alle logica
    config.py             paden, niveaus, typen, synoniemen
    data_cleaning.py      inlezen en opschonen van de drie sheets
    text.py               Nederlandse NLP pipeline
    content.py            TF-IDF en cosine similarity
    collaborative.py      item KNN, SVD, populariteit
    features.py           studentprofiel en kenmerken per (student, materiaal)
    hybrid.py             het hybride model (random forest) en het aanbevelen
    explain.py            uitleg in begrijpelijk Nederlands
    evaluation.py         tijdsplitsing, ranking metrics, baselines
    train.py              alles in een keer: opschonen, evalueren, trainen, opslaan
  api/                    FastAPI service voor de koppeling met Fastify
  tests/                  pytest tests voor cleaning, model en API
  docs/
    ai-verantwoording.md  probleem, is AI nodig, AI Act, privacy en ethiek
    koppeling-fastify.md  hoe de Fastify backend het model aanroept
```

## Installeren

Je hebt Python 3.10 of hoger nodig.

```bash
# In de map smart-reading-coach-ai
python -m venv .venv
.venv\Scripts\activate          # macOS of Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Zet de dataset van Brightspace in `data/raw/leescatalogus_recommender.xlsx` (staat er al als je deze map net hebt gekregen).
De map `data/` staat in `.gitignore`: de data van de opdrachtgever komt niet op GitHub.

## Gebruiken

```bash
# Optie 1: de notebooks (aanbevolen om de workshops te doorlopen)
jupyter lab notebooks

# Optie 2: alles in een keer vanaf de commandline
python -m leescoach.train

# Tests
pytest

# API starten
uvicorn api.main:app --reload --port 8000
```

`python -m leescoach.train` schoont de data op, evalueert alle methoden, traint het eindmodel en schrijft
`models/hybride_model.joblib` en `models/metrics.json`. De API documentatie staat daarna op `http://localhost:8000/docs`.

## Draaien met Docker

De service bouwt en draait zichzelf, los van de backend. Het model wordt getraind terwijl de image gebouwd wordt, uit
`data/raw/leescatalogus_recommender.xlsx`.

```bash
copy .env.example .env            # zet AI_API_KEY op dezelfde waarde als in de .env van de backend
docker compose up -d --build      # de service draait op http://localhost:8000
```

Zet in de `.env` van de backend `AI_SERVICE_URL=http://localhost:8000`. Na een wijziging in de code of de data:
`docker compose up -d --build` opnieuw.

## Deployen op Render

De service is een eigen Render web service (Docker), beschreven in `render.yaml`. De dataset staat niet op GitHub. Render
krijgt hem als **secret file**: een privé bestand dat alleen bij deze service hoort. Secret files moeten tekst zijn en
mogen samen maximaal 1 MB zijn, daarom gaat de xlsx er als base64 tekst in (ongeveer 280 KB). De `Dockerfile` zet die
tijdens het bouwen terug naar xlsx, traint het model en verwijdert de ruwe dataset weer uit de image.

1. **Base64 bestand maken** (PowerShell, in deze map). Het bestand komt in `data/raw` en staat dus niet in git:
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("$PWD\data\raw\leescatalogus_recommender.xlsx")) | Set-Content -NoNewline data\raw\leescatalogus_recommender.xlsx.b64
   ```
2. **Render.** Kies New, Blueprint en koppel deze repository. Render maakt de service `smart-reading-coach-ai` en een
   willekeurige `AI_API_KEY`. De eerste build mislukt nog, want de dataset ontbreekt.
3. **Secret file.** Open de service, Environment, Secret Files, Add file. Naam: `leescatalogus_recommender.xlsx.b64`.
   Plak de inhoud van het bestand uit stap 1 en kies *Save, rebuild and deploy*.
4. **Deploy hook en API key.** Kopieer onder Settings de Deploy Hook URL. Een Render API key heb je al van de backend.
5. **GitHub.** Maak in deze repository onder Settings, Environments een environment `production` met de secrets
   `RENDER_DEPLOY_HOOK_URL` en `RENDER_API_KEY` en de variabele `AI_URL` (bijvoorbeeld
   `https://smart-reading-coach-ai.onrender.com`).
6. **Backend koppelen.** Zet in de Render service van de backend `AI_SERVICE_URL` op het adres van deze service en
   `AI_API_KEY` op dezelfde waarde als hier (Environment, het oogje bij `AI_API_KEY`).

Elke push naar `main` draait daarna de tests, bouwt de image en zet hem live via de deploy hook. Een nieuwe dataset?
Maak stap 1 opnieuw en vervang de secret file.

Gratis services slapen na 15 minuten. De backend stuurt bij het opstarten een ping om deze service te wekken; zolang hij
nog slaapt, geven de regels van de backend het advies.

## Het model

Een **hybride recommender** zoals de opdracht als derde optie noemt:

1. **Kenmerken per (student, materiaal):** niveauverschil, past het type, past het genre, tekstovereenkomst tussen de
   interesses en het materiaal (TF-IDF), lijkt het op eerder gewaardeerd materiaal, past de leestijd, gemiddelde beoordeling,
   populariteit en een collaborative filtering score (item KNN).
2. **Random forest:** leert uit de interacties welke combinatie leidt tot *gelezen*, *geliket* of *opgeslagen*. De
   collaborative score en populariteit worden *out of fold* berekend, zodat er geen data leakage is.
3. **Sorteren en filteren:** materialen die de student al kent of die meer dan een niveau te moeilijk zijn vallen af.
4. **Variatie:** een *maximal marginal relevance* stap voorkomt dat de top 5 uit bijna dezelfde teksten bestaat. Dat past bij
   het doel van de opdrachtgever: studenten **meer en gevarieerder** laten lezen.
5. **Uitleg:** `explain.py` zet de kenmerken om in zinnen als *"Het gaat over vriendschap, iets waar je interesse in hebt"* en
   *"Het niveau (3F+) is net iets uitdagender dan jouw niveau (3F), goed om te groeien"*.

Een nieuwe student zonder geschiedenis krijgt meteen advies op basis van profiel en inhoud (cold start). Zodra er een
leeslijst is, schuift het advies mee.

## Resultaten

Evaluatie met een tijdsplitsing: per student de oudste 75% van de interacties om te trainen, de nieuwste 25% om te testen.

| Methode | Hitrate@5 | NDCG@5 | AUC test interacties |
| --- | --- | --- | --- |
| Willekeurig | 0.04 | 0.02 | 0.50 |
| Populariteit | 0.24 | 0.12 | 0.64 |
| Regels (huidige app) | 0.16 | 0.07 | 0.64 |
| Content based (TF-IDF) | 0.09 | 0.04 | 0.62 |
| Item KNN | 0.05 | 0.02 | 0.53 |
| Matrix factorisatie (SVD) | 0.10 | 0.06 | 0.60 |
| **Hybride model (random forest)** | **0.27** | **0.12** | **0.83** |

Het hybride model zet materialen die een student goed vindt duidelijk hoger dan materialen die hij afwijst (AUC 0.83 tegen
0.64 voor de huidige regels). In de top 5 uit de hele catalogus is het verschil met populariteit klein; zie notebook 06 voor
de uitleg. De actuele cijfers staan na elke training in `models/metrics.json`.

## API

| Methode | Route | Doel |
| --- | --- | --- |
| GET | `/api/health` | Draait de service en is er een model |
| POST | `/api/recommendations` | 3 tot 5 adviezen voor een leesprofiel uit de app plus de leeslijst |
| GET | `/api/model` | Welke kenmerken het zwaarst wegen (transparantie voor docenten) |
| GET | `/api/demo/students/{id}/recommendations` | Advies voor een student uit de dataset (S001 tot S100) |

Fouten zijn *problem details* (RFC 9457) met Nederlandse teksten, net als in de Fastify backend. Zet `AI_API_KEY` in `.env`
om alleen de backend toegang te geven (header `X-API-Key`). Zie [docs/koppeling-fastify.md](docs/koppeling-fastify.md).

## LU2 checklist

| Onderdeel uit de opdracht | Waar |
| --- | --- |
| Relevant probleem definiëren | `docs/ai-verantwoording.md` |
| Dataset van de opdrachtgever gebruiken | `data/raw`, notebook 01 |
| Veilige en privacyvriendelijke omgeving | Lokale venv, data niet in git, dataminimalisatie (`docs/ai-verantwoording.md`) |
| EDA: verkennen, opschonen, analyseren | Notebook 01, 02, 03 |
| Relevante variabelen kiezen | Notebook 01 (bivariaat), notebook 06 (feature importance) |
| Model trainen en optimaliseren | Notebook 06 (vergelijking en GridSearchCV), notebook 07 |
| Is AI de juiste tool? | Notebook 06 (vergelijking met regels), `docs/ai-verantwoording.md` |
| Risicoclassificatie EU AI Act | `docs/ai-verantwoording.md` |
| Ethisch kader | `docs/ai-verantwoording.md` |
| Integratie in de webapplicatie | `api/`, `docs/koppeling-fastify.md` |
| Human centered: uitleg en menselijke regie | `explain.py`, `/api/model`, notebook 07 |
| Profiel analyseren, vergelijken, 3 tot 5 adviezen, uitleg | `hybrid.py`, `explain.py`, notebook 07 |

## Beperkingen en volgende stappen

* De interacties en een deel van de catalogus zijn **gesimuleerd** (kolom `gegevensstatus`). De cijfers zeggen iets over deze
  dataset, niet over echte studenten. Een A/B test tegen de huidige regels is de volgende stap.
* Er is weinig Engels materiaal; drie studenten willen Engels lezen. Het model zegt dat eerlijk in de uitleg.
* Mogelijke verbeteringen: meertalige sentence embeddings in plaats van TF-IDF met synoniemen, een taalmodel dat de uitleg
  vloeiender maakt (de feiten blijven uit het model komen), feedback knop *"past niet bij mij"* in de app.
* Het gratis Render plan heeft weinig rekenkracht: een advies duurt daar langer dan lokaal.
