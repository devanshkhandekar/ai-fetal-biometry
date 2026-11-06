import {
  useMemo,
  useRef,
  useState,
} from "react";
import type { MouseEvent } from "react";

import {
  type CalibrationPoint,
  makeSquareRulerSpacing,
  screenToImagePoint,
  spacingFromKnownDistance,
} from "../lib/calibration";


type Props = {
  imageUrl: string;
  onApply: (spacing: {
    row_mm: number;
    column_mm: number;
    source: "ruler";
  }) => void;
};


export default function RulerCalibration({
  imageUrl,
  onApply,
}: Props) {
  const imageRef = useRef<HTMLImageElement>(null);

  const [points, setPoints] =
    useState<CalibrationPoint[]>([]);

  const [knownDistanceMm, setKnownDistanceMm] =
    useState<number>(10);

  const [squarePixelsConfirmed, setSquarePixelsConfirmed] =
    useState(false);

  const [message, setMessage] =
    useState<string | null>(null);

  const spacing = useMemo(() => {
    if (
      points.length !== 2 ||
      !Number.isFinite(knownDistanceMm) ||
      knownDistanceMm <= 0
    ) {
      return null;
    }

    try {
      return spacingFromKnownDistance(
        points[0],
        points[1],
        knownDistanceMm,
      );
    } catch {
      return null;
    }
  }, [points, knownDistanceMm]);

  function handleImageClick(
    event: MouseEvent<HTMLImageElement>,
  ) {
    if (!imageRef.current) {
      return;
    }

    try {
      const p = screenToImagePoint(
        event.clientX,
        event.clientY,
        imageRef.current,
      );

      setPoints((old) => {
        if (old.length >= 2) {
          return [p];
        }
        return [...old, p];
      });

      setMessage(null);
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Unable to read calibration point.",
      );
    }
  }

  function apply() {
    if (spacing == null) {
      setMessage("Select two valid calibration points first.");
      return;
    }

    if (!squarePixelsConfirmed) {
      setMessage(
        "Confirm equal horizontal and vertical pixel scale before applying one ruler measurement to both axes.",
      );
      return;
    }

    const result = makeSquareRulerSpacing(spacing);
    onApply(result);
    setMessage("Calibration applied.");
  }

  return (
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
      <div>
        <h3 className="font-semibold text-slate-900">
          Image ruler calibration
        </h3>

        <p className="mt-1 text-sm text-slate-500">
          Use this only when a known physical scale is visible in the exported image.
          Select two ruler points and enter their real distance in millimetres.
        </p>
      </div>

      <div className="overflow-auto rounded-lg border border-slate-200 bg-slate-950/5 p-2">
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Ultrasound calibration"
          onClick={handleImageClick}
          draggable={false}
          className="mx-auto max-h-[500px] max-w-full cursor-crosshair select-none rounded-md"
        />
      </div>

      <label className="block">
        <span className="text-sm font-medium text-slate-700">
          Known distance (mm)
        </span>

        <input
          type="number"
          step="0.1"
          min="0.1"
          value={knownDistanceMm}
          onChange={(e) =>
            setKnownDistanceMm(Number(e.target.value))
          }
          className="mt-1 block w-40 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
        />
      </label>

      <div className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">
        <div>Selected points: {points.length}/2</div>

        {spacing != null && (
          <div className="mt-1 font-medium">
            Calculated spacing: {spacing.toFixed(6)} mm/pixel
          </div>
        )}
      </div>

      <label className="flex items-start gap-2 text-sm text-slate-700">
        <input
          type="checkbox"
          checked={squarePixelsConfirmed}
          onChange={(e) =>
            setSquarePixelsConfirmed(e.target.checked)
          }
          className="mt-1"
        />

        <span>
          I confirm this exported image preserves equal horizontal and vertical pixel scale.
        </span>
      </label>

      {message && (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
          {message}
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => {
            setPoints([]);
            setMessage(null);
          }}
          className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700"
        >
          Reset points
        </button>

        <button
          type="button"
          disabled={spacing == null || !squarePixelsConfirmed}
          onClick={apply}
          className="rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          Apply calibration
        </button>
      </div>
    </section>
  );
}
