# Osnova Praktické Části

Pracovní osnova pro kapitolu 4 diplomové práce. Tento soubor je veden samostatně od `README.md` a `todos.md`, protože jde o psací a kompoziční podklad, nikoli o popis repa nebo task tracker.

## 4 Praktická část

### 4.1 Cíl experimentální části

- vymezit cíl praktické části
- určit, které modelové rodiny jsou porovnávány
- formulovat hlavní experimentální otázky

### 4.2 Softwarová architektura a vývojové prostředí

#### 4.2.1 Struktura projektu a sdílené moduly

- `src/data_loader.py`
- `src/notebook_setup.py`
- `src/tuning.py`
- `src/pipeline.py`
- `src/evaluation.py`
- `src/visualization.py`

#### 4.2.2 Použité knihovny a frameworky

- Darts
- StatsForecast / AutoARIMA
- Prophet
- PyTorch
- Nixtla / TimeGPT
- Chronos2
- GraniteTTM

#### 4.2.3 Výpočetní prostředí a praktická omezení

- lokální výpočetní prostředí
- reduced / smoke setup pro BTC
- interpretace timing metrik pouze v rámci jednoho datasetového běhu

### 4.3 Data a jejich příprava

#### 4.3.1 Přehled použitých datasetů

- `01` GDP yearly
- `02` Investments quarterly
- `03` EUR/CZK monthly
- `04` M5 daily
- `05` BTC hourly

#### 4.3.2 Rozdělení na train/test a volba forecast horizontu

- `test_periods`
- `seasonal_period`
- `cv_start_ratio`
- rozdíly mezi datasety

#### 4.3.3 Strategie transformace cílové řady

- deep learning modely: scaling
- foundation modely: raw scale
- statistické modely: dataset-level `raw/log`
- Box-Cox diagnostika jako rozhodovací mechanismus

#### 4.3.4 Kovariáty a multiseries specifika

- future covariates
- past covariates
- M5 a práce s více sériemi

### 4.4 Implementace predikčních modelů

#### 4.4.1 Statistické modely

- Holt-Winters
- AutoARIMA
- Prophet

#### 4.4.2 Modely hlubokého učení

- N-BEATS
- TiDE
- TFT

#### 4.4.3 Foundation modely

- Chronos2
- GraniteTTM
- TimeGPT

#### 4.4.4 Lokální vs. globální modely

- lokální statistické modely
- globální deep learning modely
- zero-shot foundation modely

### 4.5 Experimentální design a ladění

#### 4.5.1 Konfigurace hyperparametrů a tuning

- tuning grids
- random grid search
- interní vyhledávání parametrů u AutoARIMA

#### 4.5.2 Validace pomocí rolling backtestingu

- validation RMSE / MAPE
- role `historical_forecasts`
- rozdíl mezi validation a final test

#### 4.5.3 Finální retraining a test forecast

- jak vznikají finální predikce na testu
- jak se transformace invertují zpět na původní škálu

#### 4.5.4 Měření výpočetní náročnosti

- tuning time
- best configuration time
- interpretace runtime pouze v rámci datasetu

#### 4.5.5 Diebold-Mariano test a statistická významnost

- dedicated backtest artifacts
- pairwise comparison
- Holm correction
- situace, kdy některé páry nelze korektně porovnat

### 4.6 Výsledky po jednotlivých datasetech

#### 4.6.1 Dataset 01: USA GDP yearly

#### 4.6.2 Dataset 02: Investments quarterly

#### 4.6.3 Dataset 03: EUR/CZK monthly

#### 4.6.4 Dataset 04: M5 Walmart daily

#### 4.6.5 Dataset 05: BTC hourly

Doporučená vnitřní struktura každé datasetové podkapitoly:

- dataset-specific experiment setup
- selected preprocessing strategy
- main metrics
- DM results
- short interpretation

### 4.7 Souhrnná komparativní analýza

#### 4.7.1 Přesnost napříč modelovými rodinami

#### 4.7.2 Výpočetní náročnost

#### 4.7.3 Statistická významnost rozdílů

#### 4.7.4 Silné a slabé stránky porovnávaných přístupů

### 4.8 Omezení experimentu

- reduced BTC setup
- hardware constraints
- skipped DM pairs
- rozdíly v délce a frekvenci datasetů

## Poznámky

- Tato osnova je pracovní a může být ještě zpřesněna nebo zkrácena.
- Struktura odpovídá skutečnému stavu kódu a notebook workflow, ne obecné šabloně diplomové práce.
- Při budoucích detailních rozpadech a přípravě textu by tento soubor měl sloužit jako hlavní referenční kostra praktické části.
