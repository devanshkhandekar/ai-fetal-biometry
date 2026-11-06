import {
  REPORT_LANGUAGES,
  type ReportLanguage,
} from "../lib/reportLanguages";


type Props = {
  value: ReportLanguage;
  onChange: (language: ReportLanguage) => void;
  disabled?: boolean;
};


export default function ReportLanguagePicker({
  value,
  onChange,
  disabled = false,
}: Props) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">
        Report language
      </span>

      <select
        value={value}
        disabled={disabled}
        onChange={(e) =>
          onChange(e.target.value as ReportLanguage)
        }
        className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm disabled:opacity-50"
      >
        {REPORT_LANGUAGES.map((language) => (
          <option key={language} value={language}>
            {language}
          </option>
        ))}
      </select>
    </label>
  );
}
