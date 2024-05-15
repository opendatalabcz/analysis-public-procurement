## Analýza veřejných zakázek

Tento repozitář obsahuje zdrojové kódy ve formě jupyter notebooků, vytvořené v rámci bakalářské práce s názvem Analýza výherců veřejných zakázek.

Dále byly vytvořeny reporty, které jsou publikovány na webové stránce https://tenderdashboard.opendatalab.cz.
Tyto reporty obsahují informace o trhu s veřejnými zakázkami v České republice, včetně informací o jednotlivých zakázkách, zadavatelích a dodavatelích. Také je zde přehled o vývoji metrik natrénovaného modelu. 
Zdrojový kód těchto reportů je uložen v repozitáři 

## Data
Data pro práci byla získána z portálu [NEN](https://nen.nipez.cz/verejne-zakazky) a zpracována pomocí jazyka Python.
Nicméně vzhledem k jejich velikosti, zde nejsou uložena.

Data uložená ve složce by měly mít stukturu:
- `název_složky`
  - `address.csv`: obsahuje informace o adresách
  - `company.csv`: obsahuje informace o firmách
  - `contact_person.csv`: obsahuje informace o kontaktních osobách
  - `contracting_authority.csv`: obsahuje informace o zadavatelích
  - `offer.csv`: obsahuje informace o nabídkách
  - `public_procurement.csv`: obsahuje informace o veřejných zakázkách

Třídě `Preprocesor` z modulu `preprocessing.py` je poté parametrem předána cesta ke složce s daty.

## Instalace
Pro instalaci je potřeba si vytvořit virtuální prostředí a nainstalovat závislosti pomocí příkazu `pip install -r requirements.txt`.

## Spuštění

Pro vytvoření webové stránky je potřeba mít nainstalovan nástroj [Quarto](https://quarto.org/).
Poté je potřeba spustit notebook `jupyter notebook webpages\public_procurements.ipynb` a následně spustit příkaz `quarto render public_procurements.ipynb` v příkazové řádce. Tím se vytvoří statická webová stránka ve formátu HTML.

## Struktura repozitáře
- `notebooks`:obsahuje zdrojové kódy ve formě jupyter notebooků
  - `models`: obsahuje natrénované modely
- `webpages`: obsahuje zdrojový jupyter notebook pro webovou stránku

