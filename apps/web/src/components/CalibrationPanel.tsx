import { useMemo, useState } from "react";

import RulerCalibration from "./RulerCalibration";
import {
  type CalibrationSource,
  type PixelSpacing,
  makeDicomSpacing,
  makeManualSpacing,
} from "../lib/calibration";


type Props = {
  imageUrl?: string | null;
  initialSpacing?: PixelSpacing | null;
  dicomSpacing?: {
    row_mm: number;
    column_mm: number;
  } | null;
  onChange: (spacing: PixelSpacing | null) => void;
};


export default function CalibrationPanel({
  imageUrl,
  initialSpacing = null,
  dicomSpacing = null,
  onChange,
}: Props) {
  const [source, setSource] = useState<CalibrationSource>(
    initialSpacing?.source ?? (dicomSpacing ? "dicom" : "manual"),
  );

  const [rowMm, setRowMm] = useState<number>(
    initialSpacing?.row_mm ?? dicomSpacing?.row_mm ?? 0,
  );

  const [columnMm, setColumnMm] = useState<number>(
    initialSpacing?.column_mm ?? dicomSpacing?.column_mm ?? 0,
  );

  const [message, setMessage] = useState<string | null>(null);

  const dicomAvailable = useMemo(
    () => Boolean(dicomSpacing),
    [dicomSpacing],
  );

  function switchSource(next: CalibrationSource) {
    setSource(next);
    setMessage(null);

    if (next === "dicom" && dicomSpacing) {
      const spacing = makeDicomSpacing(
        dicomSpacing.row_mm,
        dicomSpacing.column_mm,
      );
      setRowMm(spacing.row_mm);
      setColumnMm(spacing.column_mm);
      onChange(spacing);
      return;
    }

    if (next !== "ruler") {
      onChange(null);
    }
  }

  function applyManual() {
    try {
      const spacing = makeManualSpacing(rowMm, columnMm);
      onChange(spacing);
      setMessage("Manual calibration applied.");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Invalid calibration.",
      );
    }
  }

  return (
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
      <div>
        <h3 className="font-semibold text-slate-900">
          Physical image calibration
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Physical fetal measurements require trusted millimetres-per-pixel calibration.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {(["manual", "ruler", "dicom"] as CalibrationSource[]).map((item) => (
          <button
            key={item}
            type="button"
            disabled={item === "dicom" && !dicomAvailable}
            onClick={() => switchSource(item)}
            className={`rounded-lg border px-3 py-2 text-sm font-medium ${
              source === item
                ? "border-slate-900 bg-slate-900 text-white"
                : "border-slate-300 bg-white text-slate-700"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            {item === "manual"
              ? "Manual"
              : item === "ruler"
                ? "Image ruler"
                : "DICOM"}
          </button>
        ))}
      </div>

      {source === "manual" && (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <label>
              <span className="text-sm font-medium text-slate-700">
                Row spacing (mm/pixel)
              </span>
              <input
                type="number"
                min="0"
                step="0.000001"
                value={rowMm || ""}
                onChange={(e) => setRowMm(Number(e.target.value))}
                className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </label>

            <label>
              <span className="text-sm font-medium text-slate-700">
                Column spacing (mm/pixel)
              </span>
              <input
                type="number"
                min="0"
                step="0.000001"
                value={columnMm || ""}
                onChange={(e) => setColumnMm(Number(e.target.value))}
                className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={applyManual}
            className="rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white"
          >
            Apply manual spacing
          </button>
        </div>
      )}

      {source === "ruler" && (
        imageUrl ? (
          <RulerCalibration
            imageUrl={imageUrl}
            onApply={(spacing) => {
              setRowMm(spacing.row_mm);
              setColumnMm(spacing.column_mm);
              onChange(spacing);
              setMessage("Ruler calibration applied.");
            }}
          />
        ) : (
          <div className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">
            An image preview is required for ruler calibration.
          </div>
        )
      )}

      {source === "dicom" && dicomSpacing && (
        <div className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800">
          DICOM calibration available: row {dicomSpacing.row_mm.toFixed(6)} mm/pixel,
          column {dicomSpacing.column_mm.toFixed(6)} mm/pixel.
        </div>
      )}

      {message && (
        <div className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">
          {message}
        </div>
      )}
    </section>
  );
}
