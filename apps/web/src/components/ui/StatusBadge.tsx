type Props = {
  status?: string | null;
};

export default function StatusBadge({ status }: Props) {
  const normalized = String(status || "UNKNOWN").toUpperCase();

  const className =
    normalized === "ACCEPT" || normalized === "COMPLETE"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : normalized === "REVIEW" || normalized === "RUNNING" || normalized === "QUEUED"
        ? "border-amber-200 bg-amber-50 text-amber-700"
        : normalized === "FAILED" || normalized === "REJECT"
          ? "border-red-200 bg-red-50 text-red-700"
          : "border-slate-200 bg-slate-50 text-slate-700";

  return (
    <span
      className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${className}`}
    >
      {normalized}
    </span>
  );
}
