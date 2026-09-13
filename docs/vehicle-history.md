# Plate recognition and Turkish commercial-vehicle history

The sponsor brief (`kamion-truck-appraisal-brief.md`) takes priority over the older
blueprint. Kamion serves road freight; this appraisal model is for Turkish used
tractor units, not passenger cars or every neighboring market. The plate stage
reads each detected vehicle in usable photos, but only the selected truck can
receive a valuation. Background vehicles retain separate observations.

## What works

- Each detection is cropped and sent to the configured vision backend. Plate,
  country, confidence, photo ID and vehicle box travel with the result. Hidden,
  ambiguous or low-confidence plates do not trigger lookup. Country must be
  explicitly visible; it is not guessed from the selected pricing market.
- A local JSON database of authorized records is matched by normalized plate AND
  country. Spaces/hyphens are removed, but letters are never substituted for digits.
- Records must identify the source, record ID, VIN, commercial registration and
  snapshot date. Duplicate matches, invalid records and snapshots older than 30
  days cannot affect valuation. The 30-day cutoff and 0.95 model-confidence gate
  are conservative policy choices, not measured accuracy guarantees.
- Only a match to the photographed VIN can affect the appraised truck's price.
  A plate alone is not a permanent vehicle identity. Uncertain/mismatched VINs
  leave history unapplied. Other vehicle records never change the subject price.
- Repaired accidents apply a **policy assumption** of 3% / 8% / 15% for minor /
  moderate / major history. These are NOT measured Turkish-market depreciation
  coefficients. Only the most severe repaired accident is used; repeated photos
  and multiple events are not stacked. No repair bill is subtracted again.
- The deduction multiplies the existing condition-adjusted estimate and range;
  the comparable-market baseline stays intact. The report shows the record,
  event ID, date, percentage, calculation and the uncalibrated-policy caveat.
- Total loss, salvage, flood, mileage discrepancies, or unresolved/ungraded
  accidents require manual appraisal instead of an arbitrary automatic price.
- Other events (for example maintenance or inspection records) are displayed,
  without inventing a numerical effect. Missing records never earn a premium.

## Connecting actual records

**No live SBM/TRAMER integration or credentials are bundled.** This is an importable
records snapshot, not an API pretending to query government or insurer systems.

SBM documents paid plate/chassis queries through TRAMER and SMS 5664. Its FAQ says
history is limited to insurer-submitted payments since 2003, with further gaps
and missing amounts. Absence of a record is therefore not an accident-free
certificate. No public developer API was established in this implementation.
Kamion must arrange authorized access with a provider, or supply authorized
vehicle reports normalized into the schema below. Do not scrape a login/payment
flow or synthesize records from model memory. A production integration also needs
market-specific validation and calibrated price effects.

Sources checked September 13, 2026:
- https://www.kamion.co/en/
- https://www.sbm.org.tr/tr/sss/?cat=10
- https://mkt.sbm.org.tr/tr/tramer-ile-sorgulama

The history feature is **off by default** to preserve existing demo inference
and pricing. Enable it explicitly with `KAMION_HISTORY_ENABLED=1` after testing
your backend, records and pricing assumptions.

Set `KAMION_HISTORY_DB=/absolute/path/vehicle-history.json` in `.env` and restart.
The local filename `/vehicle-history.json` is gitignored. Keep real records and
credentials out of the repository. Only vehicle fields are rendered, not owners.

Schema example below is **synthetic**, not a record about an actual vehicle.
Dates must describe the real source snapshot; do not refresh dates without
refreshing the records. `repaired` and severity must come from documented evidence,
not guesses from the nominal amount of a historical insurance payment.

```json
{
  "vehicles": [{
    "plate": "34 ABC 123",
    "country": "TR",
    "vin": "NM0TEST1234567891",
    "make": "Ford",
    "model": "F-MAX",
    "commercial": true,
    "source": "SYNTHETIC EXAMPLE ONLY",
    "record_id": "example-001",
    "as_of": "2026-09-13",
    "events": [{
      "id": "example-event-001",
      "date": "2025-01-02",
      "type": "accident",
      "severity": "moderate",
      "repaired": true,
      "description": "Synthetic repaired collision for illustrating the schema"
    }]
  }]
}
```

Neighboring countries use their own country-keyed records; automatic history
deductions currently apply only to TR registrations. This does not imply
SBM coverage there, support for every plate alphabet, or a calibrated foreign
valuation model. The initial transcription validator accepts Latin letters and
numbers only. Unreadable plates need a clearer plate photo; missing VIN matches
need a chassis/VIN photo and verification against the source report.

Validation: `python -m unittest tests.test_history` (synthetic records and mocked
vision responses). A real-photo end-to-end validation still requires an enabled
vision backend, representative plate photos and an authorized history source.
