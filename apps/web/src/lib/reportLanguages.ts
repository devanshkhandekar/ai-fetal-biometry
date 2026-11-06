export type ReportLanguage =
  | "English"
  | "Hindi"
  | "Gujarati"
  | "Tamil"
  | "Telugu"
  | "Arabic"
  | "French"
  | "Spanish";

export const REPORT_LANGUAGES: ReportLanguage[] = [
  "English",
  "Hindi",
  "Gujarati",
  "Tamil",
  "Telugu",
  "Arabic",
  "French",
  "Spanish",
];

export function isRtlLanguage(language: ReportLanguage): boolean {
  return language === "Arabic";
}
