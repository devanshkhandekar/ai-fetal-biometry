import { useState } from "react";

import ReportLanguagePicker from "./ReportLanguagePicker";
import {
  isRtlLanguage,
  type ReportLanguage,
} from "../lib/reportLanguages";


type Props = {
  explanation?: string | null;
  disabled?: boolean;
  onGenerate: (language: ReportLanguage) => Promise<void> | void;
};


export default function PatientExplanationPanel({
  explanation = null,
  disabled = false,
  onGenerate,
}: Props) {
  const [language, setLanguage] =
    useState<ReportLanguage>("English");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function generate() {
    try {
      setLoading(true);
      setError(null);
      await onGenerate(language);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to generate explanation.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
      <div>
        <h3 className="font-semibold text-slate-900">
          Patient-friendly explanation
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          The explanation is narrative only. It does not change measurements, statuses, or EFW.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <ReportLanguagePicker
          value={language}
          onChange={setLanguage}
          disabled={disabled || loading}
          label="Explanation language"
        />

        <button
          type="button"
          disabled={disabled || loading}
          onClick={generate}
          className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Generating…" : "Explain report"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {explanation && (
        <div
          dir={isRtlLanguage(language) ? "rtl" : "ltr"}
          className="whitespace-pre-wrap rounded-lg bg-slate-50 p-4 text-sm leading-6 text-slate-800"
        >
          {explanation}
        </div>
      )}

      <p className="text-xs text-slate-500">
        AI-assisted explanation only. A qualified clinician must review the medical report.
      </p>
    </section>
  );
}
