# Smart Reading Coach AI

Dit is het AI deel (LU2) van **Vrij lezen op maat**. Op basis van het leesprofiel en de leesgeschiedenis van een MBO of NT2
student geeft het model **3 tot 5 leesadviezen**, elk met een kort verhaaltje van Bo waarom het boek past. Het model draait
als losse FastAPI service. De Fastify backend (`smart-reading-coach-api`) roept die service aan en valt terug op zijn eigen
regels als de service niet bereikbaar is. De catalogus leest het model uit dezelfde MongoDB als de backend, zodat het
altijd precies de titels kent die de app laat zien.

Inhoud: [Notebooks](#notebooks) · [Structuur](#mappenstructuur) · [Installeren](#installeren) · [Gebruiken](#gebruiken) ·
[MongoDB](#mongodb-en-hertrainen) · [Docker](#draaien-met-docker) · [Render](#deployen-op-render) · [Het model](#het-model) · [Resultaten](#resultaten) ·
[API](#api) · [LU2 checklist](#lu2-checklist) · [Beperkingen](#beperkingen-en-volgende-stappen)

## Notebooks

Ik heb de workshops uit de LU2 repository ([AvansICT/SE-2.1-2.2-LU2](https://github.com/AvansICT/SE-2.1-2.2-LU2)) en de
relevante workshops van vorig jaar ([AvansICT/ICT1.4-AI-System-Integration-Datascience](https://github.com/AvansICT/ICT1.4-AI-System-Integration-Datascience))
toegepast op de dataset van de opdrachtgever. De notebooks bouwen op elkaar voort, dus ze zijn bedoeld om op volgorde te lezen.
De uitleg in de notebooks is Nederlands, de code is Engels.

| Notebook | Workshop | Inhoud |
| --- | --- | --- |
| `01_data_analyse_en_cleaning` | LU2 01, vorig jaar 01, 02, 03 | Missende waarden, duplicaten, datatypes, uitschieters, univariate en bivariate analyse, chi-kwadraat, betrouwbaarheidsinterval, cleaning |
| `02_dimensiereductie_en_clustering` | LU2 02, 03 en K-means uit 04 | PCA, t-SNE en UMAP op tekst en kenmerken, parameters variëren, studentsegmenten met K-means en silhouette score |
| `03_nlp_tekstverwerking` | LU2 05 | Nederlandse NLP pipeline: normaliseren, tokenizen, stopwoorden, stemming, Bag of Words, TF-IDF, interesses uitbreiden met verwante thema's |
| `04_content_based_recommender` | LU2 06 | Cosine similarity, "lijkt op", zoeken op tekst, het profiel als zoekvraag |
| `05_collaborative_filtering` | Opdracht LU2 (KNN, matrix factorisatie) | Student x materiaal matrix, item KNN, SVD met fold in, tijdsplitsing |
| `06_hybride_model_training_en_evaluatie` | LU2 04, vorig jaar 04, 05, 06 | Logistische regressie, beslisboom, random forest, gradient boosting en neuraal netwerk vergeleken, GridSearchCV, feature importance, kalibratie, baselines, eerlijkheid per groep |
| `07_eindmodel_aanbevelingen_en_uitleg` | Opdracht LU2 | Adviezen met uitleg, cold start, wat de docent ziet |

De workshops over object detectie (YOLO) heb ik niet gebruikt, omdat er in deze opdracht geen afbeeldingen zitten.

## Mappenstructuur

```
smart-reading-coach-ai/
  data/
    raw/                  dataset van de opdrachtgever (niet in git)
    processed/            opgeschoonde csv bestanden (gegenereerd)
  models/                 getraind model en metrics.json (gegenereerd)
  notebooks/              de workshops toegepast op de dataset
  reports/figures/        grafieken uit de notebooks
  src/reading_coach/      het Python package met alle logica
    config.py             paden en een paar vaste instellingen
    vocabulary.py         types, genres, talen en interactiescores, geleerd uit de data
    synonyms.py           synoniemen van interesses, geleerd uit de Open Multilingual Wordnet
    sources.py            de catalogus uit MongoDB lezen en de synoniemen daar opslaan
    data_cleaning.py      inlezen en opschonen van de drie sheets
    text.py               Nederlandse NLP pipeline
    content.py            TF-IDF, cosine similarity en het uitbreiden van interesses
    collaborative.py      item KNN, SVD en populariteit
    features.py           studentprofiel en kenmerken per (student, materiaal)
    hybrid.py             het hybride model (random forest) en het aanbevelen
    explain.py            de redenen en het verhaaltje van Bo
    evaluation.py         tijdsplitsing, ranking metrics en baselines
    train.py              alles achter elkaar: opschonen, evalueren, trainen, opslaan
  api/                    FastAPI service voor de koppeling met Fastify
    training.py           hertrainen terwijl de service draait
  tests/                  pytest tests voor cleaning, model en API
  docs/
    ai-verantwoording.md  probleem, is AI nodig, AI Act, privacy en ethiek
    koppeling-fastify.md  hoe de Fastify backend het model aanroept
```

## Installeren

Python 3.10 of hoger.

```bash
# In de map smart-reading-coach-ai
python -m venv .venv
.venv\Scripts\activate          # macOS of Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m reading_coach.synonyms    # de wordnet data voor de synoniemen (ongeveer 40 MB, eenmalig)
```

De dataset van Brightspace hoort in `data/raw/leescatalogus_recommender.xlsx`. De mappen `data/raw` en `data/processed`
staan in `.gitignore`, zodat de data van de opdrachtgever niet op GitHub komt.

## Gebruiken

```bash
jupyter lab notebooks             # de notebooks
python -m reading_coach.train     # alles in een keer vanaf de commandline
pytest                            # tests
uvicorn api.main:app --reload --port 8000
```

`python -m reading_coach.train` schoont de data op, evalueert alle methoden, traint het eindmodel en schrijft
`models/hybrid_model.joblib` en `models/metrics.json`. Met `--skip-evaluation` slaat hij de evaluatie over (dat doet de
Docker build), met `--catalog xlsx` negeert hij MongoDB. De API documentatie staat op `http://localhost:8000/docs`.

## MongoDB en hertrainen

De backend importeert de catalogus in MongoDB (`npm run import:catalog`) en bewaart bij elk boek de originele rij uit de
spreadsheet in het veld `sources`. Met `MONGO_URL` in `.env` leest het model de catalogus daaruit in plaats van uit de xlsx.
De cleaning is hetzelfde: uit MongoDB komen precies dezelfde materialen als uit de sheet (daar is een test voor).
Studenten en interacties staan alleen in de dataset van de opdrachtgever en blijven daar vandaan komen.

MongoDB wordt alleen gelezen, met één uitzondering: na elke training schrijft het model de geleerde synoniemen naar de
collectie `topic_synonyms`. De backend gebruikt die ook in zijn regels.

Het model traint opnieuw:

* **na elke catalogusimport:** de backend roept `POST /api/admin/retrain` aan als `AI_SERVICE_URL` in zijn `.env` staat;
* **bij het opstarten:** het model uit de Docker image is getraind op de sheet in de xlsx, want de build kan niet bij
  MongoDB. Met `MONGO_URL` traint de service bij het starten daarom nog een keer op de catalogus uit MongoDB
  (uit te zetten met `TRAIN_ON_START=false`).

Trainen duurt een paar seconden. Het huidige model blijft antwoorden tot het nieuwe klaar is; mislukt de training, dan blijft
het oude model in gebruik. `GET /api/health` laat met `training` zien of er een training loopt. Op de server is de ruwe
dataset na de build weg; daar gebruikt het hertrainen de opgeschoonde studenten en interacties van de vorige training.

## Draaien met Docker

De service bouwt en draait los van de backend. Het model wordt getraind tijdens het bouwen van de image, uit
`data/raw/leescatalogus_recommender.xlsx`. Zo past het model altijd bij de scikit-learn versie in de image.

```bash
copy .env.example .env            # AI_API_KEY gelijk aan die in de .env van de backend
docker compose up -d --build      # de service draait op http://localhost:8000
```

In de `.env` van de backend staat dan `AI_SERVICE_URL=http://localhost:8000`. Om de catalogus uit de MongoDB van de
backend te lezen zet ik in de `.env` hier `MONGO_URL=mongodb://host.docker.internal:27017`. Na een wijziging in de code of de data
bouw ik opnieuw met `docker compose up -d --build`.

## Deployen op Render

De service is een eigen Render web service (Docker), beschreven in `render.yaml`. Omdat de dataset niet op GitHub staat,
krijgt Render hem als **secret file**. Secret files moeten tekst zijn en mogen samen maximaal 1 MB zijn, dus de xlsx gaat
erin als base64 tekst (ongeveer 280 KB). De `Dockerfile` zet die tijdens het bouwen terug naar xlsx, traint het model en
haalt de ruwe dataset daarna weer uit de image.

1. **Base64 bestand maken** (PowerShell, in deze map). Het bestand komt in `data/raw` en blijft dus buiten git:
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("$PWD\data\raw\leescatalogus_recommender.xlsx")) | Set-Content -NoNewline data\raw\leescatalogus_recommender.xlsx.b64
   ```
2. **Render.** New, Blueprint, deze repository koppelen. Render maakt de service `smart-reading-coach-ai` en een
   willekeurige `AI_API_KEY`. De eerste build faalt, want de dataset ontbreekt nog.
3. **Secret file.** Service, Environment, Secret Files, Add file met de naam `leescatalogus_recommender.xlsx.b64` en de
   inhoud van het bestand uit stap 1. Daarna *Save, rebuild and deploy*.
4. **Deploy hook en API key.** Onder Settings staat de Deploy Hook URL. De Render API key is dezelfde als bij de backend.
5. **GitHub.** In deze repository onder Settings, Environments een environment `production` met de secrets
   `RENDER_DEPLOY_HOOK_URL` en `RENDER_API_KEY` en de variabele `AI_URL`
   (bijvoorbeeld `https://smart-reading-coach-ai.onrender.com`).
6. **Backend koppelen.** In de Render service van de backend `AI_SERVICE_URL` op het adres van deze service en
   `AI_API_KEY` op dezelfde waarde als hier.
7. **MongoDB.** `MONGO_URL` in deze service krijgt dezelfde Atlas connection string als de backend. In Atlas moet onder
   Network Access verkeer van Render toegestaan zijn (bij de backend is dat al zo).

Elke push naar `main` draait de tests, bouwt de image en zet hem live via de deploy hook. Bij een nieuwe dataset maak ik
stap 1 opnieuw en vervang ik de secret file.

Gratis services slapen na 15 minuten. De backend stuurt bij het opstarten een ping om deze service te wekken; zolang hij
nog slaapt, geven de regels van de backend het advies.

## Het model

Een **hybride recommender**, de derde optie uit de opdracht:

1. **Woordenschat uit de data.** Welke materiaaltypes, genres en talen er zijn, en hoe positief elk soort interactie is,
   staat niet in de code. Dat leert `vocabulary.py` uit de catalogus en de interacties. Bij *geliket* hoort gemiddeld een 5,
   bij *afgewezen* een 2; alles met een gemiddelde van 3,5 of hoger telt als positief. Komt er een nieuw type of genre in
   de catalogus, dan hoeft er niets aan de code te veranderen.
2. **Kenmerken per (student, materiaal):** niveauverschil, past het type, past het genre, tekstovereenkomst tussen de
   interesses en het materiaal (TF-IDF), lijkt het op eerder gewaardeerd materiaal, past de leestijd, gemiddelde beoordeling,
   populariteit en een collaborative filtering score (item KNN).
3. **Interesses uitbreiden.** Een student zegt *kunstmatige intelligentie*, de catalogus zegt *AI*. Ik breid een interesse
   op twee manieren uit, allebei zonder lijst in de code:
   * **synoniemen** uit de Open Multilingual Wordnet (het Nederlandse deel komt uit Open Dutch WordNet): thema's met
     dezelfde betekenis of een nauwere (*misdaad*: *cybercrime*). Ook de Engelse woorden tellen mee, want de catalogus
     gebruikt leenwoorden. Alleen thema's die echt in de catalogus staan blijven over;
   * **thema's die samen voorkomen:** thema's die duidelijk vaker voorkomen bij materialen over die interesse (lift).

   Bij een nieuwe catalogus worden beide opnieuw geleerd.
4. **Random forest.** Leert uit de interacties welke combinatie van kenmerken tot een positieve interactie leidt. De
   collaborative score en populariteit worden *out of fold* berekend, zodat er geen data leakage is.
5. **Sorteren en filteren.** Materialen die de student al kent of die meer dan een niveau te moeilijk zijn vallen af.
6. **Variatie.** Een *maximal marginal relevance* stap zorgt dat de top 5 niet uit bijna dezelfde teksten bestaat. Dat past
   bij het doel van de opdrachtgever: studenten **meer en gevarieerder** laten lezen.
7. **Uitleg.** `explain.py` maakt van dezelfde kenmerken een verhaaltje in de stem van Bo, bijvoorbeeld:
   > Het gaat over vriendschap, en daar lees jij graag over. Het is een boek op 2F, precies jouw niveau, en al is het
   > langer dan je wilde, in ongeveer 4 keer heb je het uit.

   Elke zin komt uit een kenmerk dat het model echt gebruikte, dus de uitleg klopt altijd met de reden. De losse redenen
   blijven ook beschikbaar, voor de docent.

Een nieuwe student zonder geschiedenis krijgt meteen advies op basis van profiel en inhoud (cold start). Zodra er een
leeslijst is, schuift het advies mee.

## Resultaten

Evaluatie met een tijdsplitsing: per student de oudste 75% van de interacties om te trainen, de nieuwste 25% om te testen.

| Methode | Hitrate@5 | NDCG@5 | AUC test interacties | AUC per student |
| --- | --- | --- | --- | --- |
| Random | 0.04 | 0.02 | 0.50 | 0.48 |
| Popularity | 0.24 | 0.12 | 0.64 | 0.65 |
| Rules (first version of the app) | 0.17 | 0.07 | 0.64 | 0.61 |
| Content based (TF-IDF) | 0.07 | 0.03 | 0.61 | 0.61 |
| Item KNN | 0.04 | 0.01 | 0.54 | 0.54 |
| Matrix factorisation (SVD) | 0.16 | 0.08 | 0.61 | 0.58 |
| **Hybrid model (Random Forest)** | **0.32** | **0.16** | **0.82** | **0.77** |

Het hybride model zet materialen die een student goed vindt duidelijk hoger dan materialen die hij afwijst (AUC 0.82 tegen
0.64 voor de regels) en wint ook in de top 5 uit de hele catalogus. De synoniemen uit de wordnet brachten de hitrate@5 van
0.28 naar 0.32, voor een iets lagere AUC (0.83 naar 0.82). De student ziet alleen de top 5, dus ik houd ze. In notebook 06
staat de uitleg bij deze cijfers. Na elke training staan de actuele cijfers in `models/metrics.json`.

## API

| Methode | Route | Doel |
| --- | --- | --- |
| GET | `/api/health` | Draait de service, is er een model (`ok` of `no_model`) en loopt er een training (`training`) |
| POST | `/api/recommendations` | 3 tot 5 adviezen voor een leesprofiel uit de app plus de leeslijst |
| GET | `/api/model` | Welke kenmerken het zwaarst wegen, plus de types en genres die het model kent |
| GET | `/api/demo/students/{id}/recommendations` | Advies voor een student uit de dataset (S001 tot S100) |
| POST | `/api/admin/retrain` | Een nieuw model trainen op de huidige catalogus (202, of 409 als er al een training loopt) |

Fouten zijn *problem details* (RFC 9457) met Nederlandse teksten, net als in de Fastify backend. Met `AI_API_KEY` in `.env`
heeft alleen de backend toegang (header `X-API-Key`). Zie [docs/koppeling-fastify.md](docs/koppeling-fastify.md).

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

* De interacties en een deel van de catalogus zijn **gesimuleerd** (kolom `gegevensstatus`). De cijfers zeggen iets over
  deze dataset, niet over echte studenten. In de app houd ik al bij of een boek via het advies of via de catalogus op de
  leeslijst komt, zodat ik het advies later met echte data kan vergelijken met de regels.
* Er is weinig Engels materiaal, terwijl drie studenten Engels willen lezen. Het model zegt dat eerlijk in de uitleg.
* De wordnet kent niet alles. Samenstellingen als *persoonlijke ontwikkeling* staan er niet in; daar helpt alleen de
  uitbreiding uit de catalogus. Meertalige sentence embeddings zouden ook die betekenis vinden, maar dat is een groot
  taalmodel als extra afhankelijkheid.
* De wordnet data valt onder de licenties van de Open Multilingual Wordnet (Open Dutch WordNet: CC BY-SA 4.0). Ik gebruik
  hem alleen tijdens het trainen; de app krijgt alleen de geleerde koppelingen.
* Het gratis Render plan heeft weinig rekenkracht, dus een advies duurt daar langer dan lokaal.
