# Osnova Praktické Části

Pracovní osnova pro kapitolu 4 diplomové práce. Tento soubor je veden samostatně od `README.md` a `todos.md`, protože jde o psací a kompoziční podklad, nikoli o popis repa nebo task tracker.

Cíl této osnovy:

- mít jednu detailní kostru pro psaní praktické části
- neztratit žádné důležité metodické ani implementační rozhodnutí
- udržet návaznost mezi kódem, notebooky, výsledky a finálním textem

## Obecná psací pravidla

- Každý graf a každá tabulka použitá v textu musí mít vlastní interpretaci.
- Každé metodické rozhodnutí musí mít stručné zdůvodnění, ne pouze popis toho, že bylo provedeno.
- Výklad má být veden od architektury a dat směrem k modelům, evaluaci a výsledkům.
- Je vhodné průběžně odlišovat:
  - co bylo součástí EDA a diagnostiky
  - co bylo součástí finální forecast pipeline
  - co je model-selection logika
  - co je inferenční / statistické porovnání
- U všech tabulek a grafů je potřeba vysvětlit:
  - co zobrazují
  - z jaké fáze experimentu pocházejí
  - jak se mají interpretovat
  - co z nich naopak vyvozovat nelze

## Doporučený způsob psaní

- Nejprve dopsat kapitoly `4.1` až `4.5`, protože jsou společné pro všechny datasety.
- Poté zpracovat `4.6` po datasetech ve stejném pořadí jako notebooky `01–05`.
- Nakonec napsat `4.7` a `4.8`, tedy souhrnnou interpretaci a omezení.

## Doporučené pracovní bloky pro generování textu

Pro účely postupného psaní a generování detailního textu je vhodné praktickou část rozdělit do těchto bloků:

### Blok A: společný rámec experimentu

- `4.1 Cíl experimentální části`
- `4.2 Softwarová architektura a vývojové prostředí`

Smysl bloku:

- ukotvit, co je cílem benchmarku
- vysvětlit architekturu repa a sdílené workflow
- popsat výpočetní prostředí a praktická omezení

### Blok B: data a příprava dat

- `4.3 Data a jejich příprava`

Smysl bloku:

- představit datasety
- oddělit preprocessing, EDA a finální forecast pipeline
- vysvětlit train/test split, transformace a covariáty

### Blok C: modely, tuning a evaluace

- `4.4 Implementace predikčních modelů`
- `4.5 Experimentální design a ladění`

Smysl bloku:

- popsat jednotlivé modelové rodiny
- vysvětlit tuning, rolling validation, final retraining a DM test
- připravit metodický základ pro interpretaci výsledků

### Blok D: datasetové výsledky

- `4.6.1` až `4.6.5`

Smysl bloku:

- každému datasetu dát samostatný, ale strukturálně stejný výklad
- přímo navázat na konkrétní notebooky, grafy a tabulky
- umožnit případné pozdější přepsání jen jedné datasetové podkapitoly bez zásahu do zbytku textu

### Blok E: souhrn a omezení

- `4.7 Souhrnná komparativní analýza`
- `4.8 Omezení experimentu`

Smysl bloku:

- uzavřít benchmark napříč datasety
- pojmenovat silné a slabé stránky modelových rodin
- oddělit metodická omezení od infrastrukturních a praktických omezení

## 4 Praktická část

### 4.1 Cíl experimentální části

Co má kapitola obsahovat:

- stručné navázání na teoretickou část
- vysvětlení, proč je praktická část postavena jako benchmark více modelových rodin
- vymezení hlavního cíle:
  - vytvořit jednotný experimentální rámec
  - porovnat statistické, deep learning a foundation modely
  - vyhodnotit je z hlediska přesnosti, výpočetní náročnosti a statistické významnosti rozdílů
- vymezení dílčích cílů:
  - sjednotit datový tok a software architekturu
  - nastavit konzistentní train/test a validation workflow
  - porovnat modely na datasetech s různou frekvencí a strukturou
  - oddělit model selection od statistického inferenčního testování

Co explicitně zmínit:

- že práce nesleduje jen „nejlepší RMSE“, ale i robustnost workflow a interpretovatelnost výsledků
- že benchmark zahrnuje malé makroekonomické řady i velké high-frequency řady
- že součástí cíle je i metodická unifikace různých modelových rodin v jednom prostředí

### 4.2 Softwarová architektura a vývojové prostředí

#### 4.2.1 Struktura projektu a sdílené moduly

Co má být popsáno:

- rozdělení projektu na:
  - preprocessing notebooky
  - exploratory data analysis notebooky
  - forecasting notebooky
  - sdílené moduly v `src/`
- důvod, proč nebyla logika ponechána jen v noteboocích
- přínos oddělení mezi řídicí vrstvou notebooku a sdílenou implementací

Co konkrétně zmínit:

- `src/data_loader.py`
  - načtení datasetu
  - rozdělení train/test
  - příprava škálovaných variant a covariát
- `src/tuning.py`
  - rolling validation
  - grid/random search
  - výběr vítězných konfigurací
- `src/pipeline.py`
  - finální retraining
  - final forecast
  - foundation modely
- `src/evaluation.py`
  - DM backtest
  - pairwise porovnání
  - shortlist logika
- `src/visualization.py`
  - grafy
  - tabulky
  - export PNG
- `src/notebook_setup.py`
  - společné importy a sdílené notebook helpery

Interpretace:

- architektura není jen technické rozdělení souborů
- je to nástroj metodické konzistence napříč datasety

#### 4.2.2 Použité knihovny a frameworky

Co pokrýt:

- Darts jako sjednocující framework
- StatsForecast / AutoARIMA
- Prophet
- PyTorch a PyTorch Lightning
- Nixtla / TimeGPT
- Hugging Face závislosti pro foundation modely
- Chronos2
- GraniteTTM
- pomocné knihovny:
  - pandas
  - NumPy
  - scikit-learn
  - Plotly
  - matplotlib

Co vysvětlit:

- proč byl zvolen Python a ne kombinace více nástrojů
- proč Darts dává smysl jako společná abstrakce
- kde Darts pomáhá a kde už je potřeba vlastní glue code

#### 4.2.3 Výpočetní prostředí a praktická omezení

Co popsat:

- lokální výpočetní prostředí
- práce v Jupyter noteboocích
- virtuální prostředí a dependency management
- praktická omezení běhu na lokálním hardware

Co explicitně zmínit:

- timing metriky mají být interpretovány uvnitř datasetu, ne napříč datasety
- `05` BTC je ve finální podobě veden jako reduced / smoke run nad nejnovějšími `50k` body
- důvodem není metodická libovůle, ale praktická neproveditelnost plného běhu zejména kvůli AutoARIMA rolling validaci
- externí API služby přinášejí:
  - měsíční limity
  - minutové rate limity
  - síťovou nestabilitu
  - provider-side chyby
- u `TimeGPT` byl zaveden konzervativní throttling / retry mechanismus

Co je dobré zmínit jako zajímavost:

- `04` M5 je z hlediska API a stability běhu prakticky náročnější než `05` BTC, i když BTC je větší dataset
- důvodem je hustý rolling multiseries backtest a počet samostatných forecast callů

### 4.3 Data a jejich příprava

#### 4.3.1 Přehled použitých datasetů

U každého datasetu uvést:

- název datasetu
- doménu
- frekvenci
- cílovou proměnnou
- smysl zařazení do benchmarku

Konkrétně:

- `01` USA real GDP yearly
  - malá roční makroekonomická řada
- `02` investments quarterly
  - čtvrtletní makroekonomická řada
- `03` EUR/CZK monthly
  - měsíční kurzová řada
- `04` M5 Walmart daily
  - denní retailová multiseries data
- `05` BTC hourly
  - hodinová volatilní high-frequency finanční řada

Interpretace:

- datasety nejsou vybrány náhodně
- mají pokrýt různé frekvence, délky, sezónnost, volatilitu i strukturální vlastnosti

#### 4.3.2 Předzpracování dat v preprocessing noteboocích

Co popsat:

- načtení surových dat
- čištění
- převod timestampů
- výběr sloupců
- přípravu finálních benchmark CSV

Co zmínit:

- že preprocessing je dataset-specific
- ale výstupem je sjednocený formát vhodný pro sdílený loader
- že multiseries M5 vyžaduje jinou přípravu než single-series datasety

#### 4.3.3 Exploratory data analysis a diagnostika řad

Co zahrnout:

- vizuální průzkum dat
- trend
- sezónnost
- volatilita
- strukturální rozdíly mezi datasety
- ADF testy

Důležitá interpretace:

- ADF testy jsou diagnostická součást EDA
- nejsou to globální produkční pravidla forecast pipeline
- neslouží jako přímý spouštěč pro transformaci všech modelů

Co explicitně vysvětlit:

- rozdíl mezi:
  - diagnostikou stacionarity
  - stabilizací variance
- tedy proč se v EDA objevuje ADF, ale ve forecast pipeline Box-Cox/log rozhodování

#### 4.3.4 Rozdělení na train/test a volba forecast horizontu

Co popsat:

- definici `test_periods`
- význam `seasonal_period`
- význam `cv_start_ratio`
- rozdílnou délku train/test napříč datasety

Co interpretovat:

- proč je forecast horizon u každého datasetu jiný
- proč denní a hodinové řady vyžadují jinou validační strategii než roční řady
- že finální test data modely při tuningu „nevidí“

Artefakty:

- train/test split graf

K tomu vysvětlit:

- co je train
- co je test
- proč je ten split důležitý pro validní benchmark

#### 4.3.5 Strategie transformace cílové řady

Co pokrýt:

- deep learning modely:
  - scaling
- foundation modely:
  - raw scale
- statistické modely:
  - dataset-level `raw/log`

Co vysvětlit detailně:

- proč nebyla zvolena jedna globální transformace pro všechny modely
- proč statistical modely dostaly dataset-level rozhodnutí
- jak funguje Box-Cox diagnostika
- co znamená `lambda`
- co znamená CI kolem `lambda`
- jak se rozhoduje mezi `raw` a `log`

Nutné zdůraznit:

- Box-Cox diagnostika slouží jako rozhodovací mechanismus
- v benchmarku se nepoužívá plný Box-Cox transform
- výsledná volba je zjednodušená na `raw` nebo `log`
- validační výstupy, dedicated backtest i finální forecast jsou po inferenci převáděny zpět na původní škálu cílové proměnné, aby byly všechny modely porovnávány na stejné interpretační bázi

#### 4.3.6 Kovariáty a multiseries specifika

Co má kapitola popsat:

- future covariates
- past covariates
- static covariates
- proč jsou relevantní hlavně u `04`

Co je dobré zmínit:

- M5 není jen „další řada“, ale strukturálně jiný problém
- některé modely pracují lokálně po sériích, jiné globálně napříč sériemi
- to má dopad jak na tuning, tak na interpretaci času i výsledků

### 4.4 Implementace predikčních modelů

#### 4.4.1 Statistické modely

U každého modelu stručně:

- proč byl zařazen
- jaká je jeho role v benchmarku
- jak byl implementačně napojen do pipeline

Konkrétně:

- Holt-Winters
  - klasický baseline se sezónností
- AutoARIMA
  - automatická selekce řádu
  - interní search, ale externě pouze jedna winning konfigurace
- Prophet
  - trendově-sezónní model s odlišnou modelovou filozofií

#### 4.4.2 Modely hlubokého učení

Co popsat:

- TiDE
- N-BEATS
- TFT

Co zdůraznit:

- jsou trénovány na škálovaných datech
- u multiseries datasetu mohou využívat globální trénink
- používají jiný fit/predict režim než statistical modely

#### 4.4.3 Foundation modely

Co pokrýt:

- Chronos2
- GraniteTTM
- TimeGPT

Co zdůraznit:

- nejde o klasické znovu plně trénované modely v tomtéž smyslu jako DL modely
- `TimeGPT` je externí API služba
- foundation modely přinášejí i infrastrukturní omezení:
  - model download
  - SSL / trust store
  - rate limits
  - provider-side chyby

#### 4.4.4 Lokální vs. globální modely

Co vysvětlit:

- lokální statistické modely
- globální DL modely
- zero-shot foundation modely

Co je dobré vypíchnout:

- nejde jen o rozdíl architektur
- ale i o rozdíl v datovém toku, validaci a interpretaci runtime

### 4.5 Experimentální design a ladění

#### 4.5.1 Konfigurace hyperparametrů a tuning

Co má být popsáno:

- grid search
- random grid search
- zjednodušené search prostory
- rozdíl mezi externím gridem a interním search mechanismem u AutoARIMA

Co nezapomenout:

- evidence vítězných konfigurací
- tabulka `Selected Model Configurations`
- že se ukládají skutečně vybrané parametry, ne jen finální názvy modelů

#### 4.5.2 Validace pomocí rolling backtestingu

Co vysvětlit:

- proč rolling validation místo jednorázového splitu
- role `historical_forecasts`
- `retrain=True` vs `retrain=False`
- `last_points_only=True`
- rozdíl mezi local a global modely

Nutná interpretace:

- validation RMSE / MAPE jsou model-selection metriky
- nejsou totožné s finálním testem

#### 4.5.3 Finální retraining a test forecast

Co popsat:

- po tuningu se model znovu sestaví s vítěznými parametry
- natrénuje se na plném train splitu
- vygeneruje se finální future/test forecast

Co vysvětlit:

- proč je tahle fáze oddělená od validation
- jak se invertují škálování a transformace
- proč některé modely mohou mít validační výsledek, ale selhat ve finálním future forecastu, pokud narazí na externí API / payload problém

#### 4.5.4 Měření výpočetní náročnosti

Co zahrnout:

- total tuning time
- best configuration training time
- interpretaci runtime v rámci datasetu

Co výslovně zmínit:

- runtime není absolutní laboratorní benchmark
- `TimeGPT` runtime zahrnuje i síťovou latenci a provider-side chování
- throttling u `TimeGPT` může prodloužit běh záměrně, aby se zvýšila stabilita experimentu

#### 4.5.5 Diebold-Mariano test a statistická významnost

Co vysvětlit:

- proč nestačí jen RMSE ranking
- proč je potřeba inferenční porovnání
- co jsou dedicated backtest artifacts
- jak vzniká shortlist modelů
- proč se DM nepočítá nad celým param gridem

Nutné metodické body:

- DM používá `criterion="mse"`
- to je konzistentní s výběrem modelů podle `RMSE`
- `RMSE` v DM backtest summary a `RMSE` v selected-config table nejsou stejné veličiny
- DM backtest summary je oddělený dedikovaný backtest pro statistické porovnání
- pairwise porovnání může být omezeno chybějícím overlapem validačních bodů

Dataset-specific praktické poznámky:

- u `04` může být použit `forecast_horizon=1`, `stride=2` jako kompromis mezi hustotou DM backtestu a API/runtime náklady
- u `05` lze oddělit:
  - backtest forecast horizon
  - `dm_h`

### 4.6 Výsledky po jednotlivých datasetech

#### Obecná doporučená struktura každé datasetové podkapitoly

Každý dataset zpracovat ve stejném pořadí:

1. dataset-specific setup
2. zvolená transformace / preprocessing
3. přehled vítězných konfigurací
4. comparison graf a hlavní metriky
5. forecast graf
6. Box-Cox / split / další podpůrné artefakty
7. DM summary, heatmap, pairwise tabulka
8. interpretace výsledků

Co nezapomenout:

- každý vložený graf/tabulku okomentovat
- uvést, co je validation výsledek a co final test výsledek
- krátce zmínit i časový / praktický kontext, pokud je důležitý

#### 4.6.1 Dataset 01: USA GDP yearly

Pokrýt:

- malá roční řada
- dopad malé délky řady na validaci
- vhodnost statistical baseline
- chování deep learning a foundation modelů v malé datové situaci

#### 4.6.2 Dataset 02: Investments quarterly

Pokrýt:

- čtvrtletní makroekonomická řada
- interpretaci selected configs vs DM backtest summary
- vysvětlit případné změny pořadí modelů mezi tuning a DM

#### 4.6.3 Dataset 03: EUR/CZK monthly

Pokrýt:

- měnový kurz
- raw/log rozhodnutí
- interpretaci volatility a obtížnosti predikce

#### 4.6.4 Dataset 04: M5 Walmart daily

Pokrýt detailně:

- multiseries charakter problému
- covariáty
- lokální vs globální modely
- praktické komplikace s foundation modely a `TimeGPT`
- API budget a throttling
- případné změny DM nastavení (`stride=2`)
- interpretaci neúplných DM párů, pokud k nim dojde

#### 4.6.5 Dataset 05: BTC hourly

Při psaní vycházet z:

- reduced / smoke varianty nad nejnovějšími `50k` body

Nutné explicitně uvést:

- že full run byl zvažován a testován
- že rozhodující praktickou překážkou byl zejména AutoARIMA rolling validační režim
- že reduced varianta není zvolena kvůli TimeGPT API budgetu, ale kvůli celkové výpočetní proveditelnosti benchmarku

Pokrýt:

- vysoká frekvence
- vysoká výpočetní náročnost
- odlišný validační protokol
- praktické kompromisy nutné pro proveditelný experiment

### 4.7 Souhrnná komparativní analýza

#### 4.7.1 Přesnost napříč modelovými rodinami

Co porovnat:

- statistical vs DL vs foundation
- malé vs velké datasety
- nízká vs vysoká frekvence

#### 4.7.2 Výpočetní náročnost

Co zdůraznit:

- některé modely jsou levné a robustní
- některé drahé, ale výkonné
- u externích API modelů je nutné chápat runtime jinak než u čistě lokálních modelů

#### 4.7.3 Statistická významnost rozdílů

Co vyložit:

- kde DM skutečně potvrdil rozdíl
- kde rozdíl nepotvrdil
- kde malý počet bodů nebo chybějící overlap omezuje interpretaci

#### 4.7.4 Silné a slabé stránky porovnávaných přístupů

Vhodné rozdělení:

- statistické modely
- deep learning modely
- foundation modely

U každé skupiny:

- přesnost
- robustnost
- nároky na data
- nároky na infrastrukturu
- reprodukovatelnost

### 4.8 Omezení experimentu

Co shrnout:

- hardware constraints
- různé délky a frekvence datasetů
- reduced BTC experiment místo plného běhu
- skipped DM pairs
- externí API limity a request budget u TimeGPT
- potřeba throttlingu / retry / SSL trust-store workaround pro některé foundation modely
- fakt, že externí API modely jsou infrastrukturně citlivější než lokální modely

Co je dobré explicitně uvést:

- ne všechna omezení jsou metodická
- část omezení je čistě praktická / infrastrukturní
- i to je ale relevantní výsledek při porovnání reálně použitelných forecasting přístupů

## Poznámky

- Tato osnova je pracovní a může být ještě zpřesněna nebo zkrácena.
- Struktura odpovídá skutečnému stavu kódu, notebook workflow a pomocných analytických notebooků.
- Při budoucích detailních rozpadech a přípravě textu by tento soubor měl sloužit jako hlavní referenční kostra praktické části.
- Při samotném psaní je lepší jít po kapitolách `4.1` -> `4.5`, pak po datasetech `4.6`, a nakonec dopsat `4.7` a `4.8`.
