# Phase 5 web/src overlay

This folder is intentionally additive. It does not replace routes, login, API clients, upload logic, inference polling, or report logic.

## Files

- `lib/calibration.ts` — calibration math and typed spacing helpers.
- `components/RulerCalibration.tsx` — two-point ruler calibration for non-DICOM exports.
- `components/CalibrationPanel.tsx` — unified Manual / Ruler / DICOM calibration UI.
- `lib/reportLanguages.ts` — supported report languages.
- `components/ReportLanguagePicker.tsx` — language selector.
- `components/PatientExplanationPanel.tsx` — UI wrapper for your existing explain/translate API callback.
- `components/EfwMethodCard.tsx` — transparent EFW display without changing the EFW calculation.
- `components/ui/StatusBadge.tsx` — reusable status presentation.
- `styles/phase5.css` — optional additive application styling.

## Minimal calibration integration

In your existing upload page/component:

```tsx
import { useState } from "react";
import CalibrationPanel from "./components/CalibrationPanel";
import type { PixelSpacing } from "./lib/calibration";

const [pixelSpacing, setPixelSpacing] = useState<PixelSpacing | null>(null);

<CalibrationPanel
  imageUrl={previewUrl}
  dicomSpacing={null}
  onChange={setPixelSpacing}
/>
```

When creating your existing upload payload, preserve your API and only map the selected spacing:

```ts
spacing: pixelSpacing
  ? {
      row_mm: pixelSpacing.row_mm,
      column_mm: pixelSpacing.column_mm,
      source: pixelSpacing.source,
    }
  : null
```

If your backend currently validates only `source: "manual"`, extend that backend enum/Literal to also allow `"ruler"` and `"dicom"` before sending those values.

## Minimal explanation integration

Keep your current API client. The component deliberately does not know your endpoint URLs.

```tsx
<PatientExplanationPanel
  explanation={explanation}
  onGenerate={async (language) => {
    // Call your existing:
    // POST /v1/reports/{report_id}/explain
    // or /translate
    // Then update explanation state from the response.
  }}
/>
```

## Minimal EFW display integration

```tsx
<EfwMethodCard
  value={efwValue}
  status={efwStatus}
  hcMm={hcValue}
  acMm={acValue}
  flMm={flValue}
  calibrationSource={spacingSource}
  formula={efwFormula}
/>
```

This is display-only. It never recalculates EFW.

## Styling

Import once from an existing root file such as `main.tsx`, `App.tsx`, or your existing CSS entry:

```ts
import "./styles/phase5.css";
```

If your current Tailwind app already has the desired shell styling, this stylesheet can be omitted entirely.

## Important safety behavior

- Do not infer anatomical mm/pixel from PNG DPI or browser dimensions.
- Ruler mode requires a known physical scale visible in the exported image.
- The current ruler implementation applies one measured scale to both axes only after explicit confirmation that the export preserves square pixels.
- If horizontal and vertical scales differ, keep manual row/column calibration until a two-axis ruler UI is added.
- The explanation component is narrative-only and must not overwrite deterministic measurements.
- `EfwMethodCard` displays the existing formula/result only; it does not change the Phase-4 calculation.
