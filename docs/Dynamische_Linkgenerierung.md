# Dynamische Link-Generierung über URL-Parameter

Dieses System unterstützt die dynamische Generierung von Links und Integrationen über URL-Parameter, insbesondere für die Anbindung an HubSpot.

## Funktionsweise

- Über den URL-Parameter `hubspot_id` kann eine eindeutige ID an die Anwendung übergeben werden, z.B.:
  
  ```
  https://<deine-app-url>?hubspot_id=12345
  ```

- Die Anwendung kann diesen Parameter nutzen, um spezifische Inhalte, Links oder Integrationen bereitzustellen.

## HubSpot-Integration

- Die Umgebungsvariable `HUBSPOT_URL` definiert die Basis-URL für HubSpot (z.B. `https://app.hubspot.com`).
- In Kombination mit dem Parameter `hubspot_id` können dynamisch Links zu HubSpot-Objekten erzeugt werden, z.B.:

  ```
  ${HUBSPOT_URL}/contacts/<hubspot_id>
  ```

- Beispiel für einen vollständigen Link:

  ```
  https://app.hubspot.com/contacts/12345
  ```

## Hinweise

- Setze die Variable `HUBSPOT_URL` in deiner `.env`-Datei entsprechend deiner HubSpot-Instanz.
- Die dynamische Link-Generierung kann für weitere Integrationen und Anwendungsfälle erweitert werden.
