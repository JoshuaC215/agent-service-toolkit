# RooMeeting

RooMeeting ist ein rein Python-basierter Meeting-Analyse-Agent, der nahtlos in das roosiAIOS-Ökosystem integriert ist. Er nutzt KI-gestützte Transkription und Extraktion, um aus aufgezeichneten Meetings wichtige Erkenntnisse wie Aufgaben, Entscheidungen, Fristen und Teilnehmer automatisch zu generieren.

## Features

- Audioaufnahme & Datei-Upload
- Automatische Transkription (OpenAI/Groq Whisper)
- Extraktion von:
  - Aufgaben (Tasks)
  - Entscheidungen (Decisions)
  - Folgeaufgaben (Follow-ups)
  - Fristen (Deadlines)
  - Teilnehmer (Attendees)
- Komplette Python-Implementierung, kompatibel mit roosiAIOS-Services

## Design

![RooMeeting UI – Notion-inspiriertes Design](media/design_notion.png)

Das Design orientiert sich an einem klaren, minimalistischen Notion-Style:
- Karten-Layout mit abgerundeten Ecken und feinem Schatten für visuelle Abgrenzung
- Obere Leiste mit Zeitstempel (`@Heute 07:59`) und Consent-Hinweis
- Zwei Tabs: **Notizen** und **Transkription** für einfachen Kontext-Wechsel
- Auffälliger Consent-Button mit rotem Status-Indikator
- Dezente Grau- und Weißtöne mit klarer Typografie-Hierarchie für Lesbarkeit

## Integration in roosiAIOS

RooMeeting lässt sich einfach als Agent in das Agent Service Toolkit von roosiAIOS einbinden. Er bietet standardisierte Schnittstellen für:
- Automatisierten Transkriptions-Workflow
- Flexible Konfiguration der KI-Modelle und Sprachausgabe
- Skalierbare Storage-Optionen für Meeting-Daten
