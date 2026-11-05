export type Point = {
  x: number;
  y: number;
};

export function pixelDistance(
  a: Point,
  b: Point,
): number {
  return Math.hypot(
    b.x - a.x,
    b.y - a.y,
  );
}

export function spacingFromCalibration(
  a: Point,
  b: Point,
  knownDistanceMm: number,
): number {
  const px = pixelDistance(a, b);

  if (!Number.isFinite(px) || px <= 0) {
    throw new Error(
      "Calibration points must be different.",
    );
  }

  if (
    !Number.isFinite(knownDistanceMm) ||
    knownDistanceMm <= 0
  ) {
    throw new Error(
      "Known calibration distance must be positive.",
    );
  }

  return knownDistanceMm / px;
}

export function displayPointToNaturalPixel(
  clientX: number,
  clientY: number,
  image: HTMLImageElement,
): Point {
  const rect = image.getBoundingClientRect();

  const scaleX =
    image.naturalWidth / rect.width;

  const scaleY =
    image.naturalHeight / rect.height;

  return {
    x:
      (clientX - rect.left) *
      scaleX,

    y:
      (clientY - rect.top) *
      scaleY,
  };
}
