## Analýza veřejných zakázek

Tento repozitář obsahuje zdrojové kódy ve formě jupyter notebooků, vytvořené v rámci bakalářské práce s názvem Analýza výherců veřejných zakázek.

Dále byly vytvořeny reporty, které jsou publikovány na webové stránce https://tenderdashboard.opendatalab.cz.
Tyto reporty obsahují informace o trhu s veřejnými zakázkami v České republice, včetně informací o jednotlivých zakázkách, zadavatelích a dodavatelích. Také je zde přehled o vývoji metrik natrénovaného modelu. 
Zdrojový kód těchto reportů je uložen v repozitáři 

## Data
Data pro práci byla získána z portálu [NEN](https://nen.nipez.cz/verejne-zakazky) a zpracována pomocí jazyka Python.
Nicméně vzhledem k jejich velikosti, zde nejsou uložena.

Data uložená ve složce musí dodržet následující stukturu:
- `data`
  - `address.csv`: obsahuje informace o adresách
  - `company.csv`: obsahuje informace o firmách
  - `contact_person.csv`: obsahuje informace o kontaktních osobách
  - `contracting_authority.csv`: obsahuje informace o zadavatelích
  - `offer.csv`: obsahuje informace o nabídkách
  - `public_procurement.csv`: obsahuje informace o veřejných zakázkách


## Instalace
Pro spuštění všech JUPYTER notebooků je potřeba si vytvořit virtuální prostředí a nainstalovat závislosti pomocí příkazu `pip install -r requirements.txt`.

## Struktura repozitáře
- `notebooks`:obsahuje zdrojové kódy ve formě jupyter notebooků
  - `models`: obsahuje natrénované modely
- `webpages`: obsahuje zdrojový jupyter notebook pro webovou stránku a soubory pro vytvoření Docker image 
  - `www`: složka s webovou stránkou, do které se automaticky vygeneruje HTML soubor
    - `images`: složka s obrázky pro webovou stránku
  - `data`: složka s daty pro webovou stránku

## Vygenerování webové stránky
Pro vygenerování webové stránky je potřeba vytvořit Docker image pomocí příkazu: `docker build --no_cache -t <název image> <cesta k Dockerfile>` a následně spustit kontejner pomocí příkazu: 
` docker run -v <cesta ke složce s daty>:/webpages/data -v <cesta ke složce pro vygenerovaný HTML soubor>:/webpages/www web_python_image`
Tento příkaz spustí Docker kontejner, který vygeneruje webovou stránku ze složky s daty a uloží ji do vybrané složky.