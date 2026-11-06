export type CalibrationPoint = {
  x: number;
  y: number;
};

export type CalibrationSource = "manual" | "ruler" | "dicom";

export type PixelSpacing = {
  row_mm: number;
  column_mm: number;
  source: CalibrationSource;
};

export function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

export function pixelDistance(
  a: CalibrationPoint,
  b: CalibrationPoint,
): number {
  return Math.hypot(
    b.x - a.x,
    b.y - a.y,
  );
}

export function spacingFromKnownDistance(
  a: CalibrationPoint,
  b: CalibrationPoint,
  knownDistanceMm: number,
): number {
  const pixels = pixelDistance(a, b);

  if (!isPositiveFinite(pixels)) {
    throw new Error("Calibration points must be different.");
  }

  if (!isPositiveFinite(knownDistanceMm)) {
    throw new Error("Known distance must be greater than zero.");
  }

  return knownDistanceMm / pixels;
}

export function screenToImagePoint(
  clientX: number,
  clientY: number,
  image: HTMLImageElement,
): CalibrationPoint {
  const rect = image.getBoundingClientRect();

  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error("Image has no rendered size.");
  }

  const scaleX = image.naturalWidth / rect.width;
  const scaleY = image.naturalHeight / rect.height;

  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

export function validatePixelSpacing<T extends PixelSpacing>(
  spacing: T,
): T {
  if (!isPositiveFinite(spacing.row_mm)) {
    throw new Error("Row pixel spacing must be greater than zero.");
  }

  if (!isPositiveFinite(spacing.column_mm)) {
    throw new Error("Column pixel spacing must be greater than zero.");
  }

  return spacing;
}

export function makeManualSpacing(
  rowMm: number,
  columnMm: number,
): PixelSpacing & { source: "manual" } {
  return validatePixelSpacing({
    row_mm: rowMm,
    column_mm: columnMm,
    source: "manual",
  });
}

export function makeDicomSpacing(
  rowMm: number,
  columnMm: number,
): PixelSpacing & { source: "dicom" } {
  return validatePixelSpacing({
    row_mm: rowMm,
    column_mm: columnMm,
    source: "dicom",
  });
}

export function makeSquareRulerSpacing(
  mmPerPixel: number,
): PixelSpacing & { source: "ruler" } {
  return validatePixelSpacing({
    row_mm: mmPerPixel,
    column_mm: mmPerPixel,
    source: "ruler",
  });
}
