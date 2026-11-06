type Props = {
  value?: number | null;
  status?: string | null;
  hcMm?: number | null;
  acMm?: number | null;
  flMm?: number | null;
  calibrationSource?: string | null;
  formula?: string | null;
};

function format(value: number | null | undefined, digits = 2): string {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

export default function EfwMethodCard({
  value,
  status,
  hcMm,
  acMm,
  flMm,
  calibrationSource,
  formula,
}: Props) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium text-slate-500">
            Estimated fetal weight
          </div>
          <div className="mt-1 text-3xl font-semibold text-slate-900">
            {format(value, 1)} g
          </div>
        </div>

        {status && (
          <span className="rounded-full border border-slate-200 px-2.5 py-1 text-xs font-semibold text-slate-700">
            {status}
          </span>
        )}
      </div>

      <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">HC input</dt>
          <dd className="font-medium text-slate-900">{format(hcMm, 3)} mm</dd>
        </div>
        <div>
          <dt className="text-slate-500">AC input</dt>
          <dd className="font-medium text-slate-900">{format(acMm, 3)} mm</dd>
        </div>
        <div>
          <dt className="text-slate-500">FL input</dt>
          <dd className="font-medium text-slate-900">{format(flMm, 3)} mm</dd>
        </div>
        <div>
          <dt className="text-slate-500">Calibration</dt>
          <dd className="font-medium text-slate-900">
            {calibrationSource || "Not recorded"}
          </dd>
        </div>
      </dl>

      <div className="mt-4 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">
        <strong>Method:</strong>{" "}
        {formula || "Hadlock HC + AC + FL"}
        <br />
        Estimated value only — clinician review is required.
      </div>
    </section>
  );
}
