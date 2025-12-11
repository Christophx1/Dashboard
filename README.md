<<<<<<< HEAD
# Stock Dashboard (Dash)

Kurzanleitung zum Einrichten und Starten der App auf Windows (PowerShell).

## Übersicht
Dieses Projekt ist ein Dash-basiertes Dashboard für Aktien/Portfolio mit persistenten Dateien im Ordner `gui/` (`portfolio.json`, `transactions.json`, `balance.json`). Die Hauptdatei mit Kontostand-Integration ist `app_dash mit Kontostand.py`.

## Voraussetzungen
- Python 3.10+ (installiert und in PATH)
- Windows PowerShell (Anweisungen unten)

## Setup (PowerShell)
1. Im Projektordner (z. B. `c:\Users\baufe\Desktop\Dash`) ein virtuelles Environment erstellen:

```powershell
python -m venv .venv
```

2. Das venv aktivieren (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Abhängigkeiten installieren:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## App starten
Nachdem das venv aktiviert ist und die Anforderungen installiert sind, starte die App:

```powershell
python "app_dash mit Kontostand.py"
```

Öffne dann im Browser: http://localhost:8050

## Wichtige Hinweise
- Persistente Daten liegen im Ordner `gui/`:
  - `portfolio.json` — gespeicherte Positionen
  - `transactions.json` — Transaktions-Log
  - `balance.json` — Kontostand

- Wenn du Änderungen am Python-Code vornimmst, stoppe den laufenden Server (Ctrl+C) und starte ihn neu, damit die Änderungen wirksam werden.

- Um die exakten, aktuell installierten Pakete in `requirements.txt` mit Versionen zu exportieren (nachdem du alles installiert hast), führe im aktiven venv aus:

```powershell
pip freeze > requirements.txt
```

## Fehlerbehebung
- Duplicate-callback-Fehler in Dash: meist verursacht durch mehrere laufende Instanzen oder mehrfach registrierte Callbacks. Stelle sicher, dass nur eine App-Instanz läuft und starte neu.
- Falls yfinance oder Netzwerk-Abfragen fehlschlagen: prüfe Internetzugang und Firewall.

## Kontakt
Wenn etwas nicht funktioniert, sende die Fehlermeldung (Stacktrace) oder die Terminal-Ausgabe hierher, dann helfe ich beim Debuggen.
=======
# Dash
>>>>>>> 5574786787d37f47e3da496452a08e7db6a79706




257777891
