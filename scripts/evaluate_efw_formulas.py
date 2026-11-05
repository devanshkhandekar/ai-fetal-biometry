#!/usr/bin/env python3

import argparse
import csv
import math
import statistics


def hadlock_hc_ac_fl(
    hc_mm,
    ac_mm,
    fl_mm,
):
    hc = hc_mm / 10.0
    ac = ac_mm / 10.0
    fl = fl_mm / 10.0

    log10_efw = (
        1.326
        - 0.00326 * ac * fl
        + 0.0107 * hc
        + 0.0438 * ac
        + 0.158 * fl
    )

    return 10 ** log10_efw


def hadlock_bpd_hc_ac_fl(
    bpd_mm,
    hc_mm,
    ac_mm,
    fl_mm,
):
    bpd = bpd_mm / 10.0
    hc = hc_mm / 10.0
    ac = ac_mm / 10.0
    fl = fl_mm / 10.0

    log10_efw = (
        1.3596
        + 0.0064 * hc
        + 0.0424 * ac
        + 0.174 * fl
        + 0.00061 * bpd * ac
        - 0.00386 * ac * fl
    )

    return 10 ** log10_efw


def intergrowth_hc_ac(
    hc_mm,
    ac_mm,
):
    hc_cm = hc_mm / 10.0
    ac_cm = ac_mm / 10.0

    x = ac_cm / 100.0
    y = hc_cm / 100.0

    ln_efw = (
        5.084820
        - 54.06633 * (x ** 3)
        - 95.80076
        * (x ** 3)
        * math.log(x)
        + 3.136370 * y
    )

    return math.exp(ln_efw)


def metrics(
    actual,
    predicted,
):
    errors = [
        p - a
        for a, p
        in zip(actual, predicted)
    ]

    abs_errors = [
        abs(v)
        for v in errors
    ]

    pct_errors = [
        abs(p - a) / a * 100.0
        for a, p
        in zip(actual, predicted)
        if a > 0
    ]

    signed_pct = [
        (p - a) / a * 100.0
        for a, p
        in zip(actual, predicted)
        if a > 0
    ]

    within_10 = sum(
        p <= 10.0
        for p in pct_errors
    ) / len(pct_errors) * 100.0

    within_15 = sum(
        p <= 15.0
        for p in pct_errors
    ) / len(pct_errors) * 100.0

    return {
        "n": len(actual),
        "bias_g":
            statistics.mean(errors),

        "mae_g":
            statistics.mean(abs_errors),

        "mape_pct":
            statistics.mean(pct_errors),

        "mean_pct_error":
            statistics.mean(signed_pct),

        "within_10_pct":
            within_10,

        "within_15_pct":
            within_15,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_path",
        help=(
            "CSV columns: "
            "hc_mm,ac_mm,fl_mm,bpd_mm,"
            "birth_weight_g"
        ),
    )

    args = parser.parse_args()

    actual = []

    predictions = {
        "hadlock_hc_ac_fl": [],
        "hadlock_bpd_hc_ac_fl": [],
        "intergrowth_hc_ac": [],
    }

    with open(
        args.csv_path,
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            try:
                hc = float(row["hc_mm"])
                ac = float(row["ac_mm"])
                fl = float(row["fl_mm"])
                bpd = float(row["bpd_mm"])
                bw = float(row["birth_weight_g"])
            except Exception:
                continue

            actual.append(bw)

            predictions[
                "hadlock_hc_ac_fl"
            ].append(
                hadlock_hc_ac_fl(
                    hc,
                    ac,
                    fl,
                )
            )

            predictions[
                "hadlock_bpd_hc_ac_fl"
            ].append(
                hadlock_bpd_hc_ac_fl(
                    bpd,
                    hc,
                    ac,
                    fl,
                )
            )

            predictions[
                "intergrowth_hc_ac"
            ].append(
                intergrowth_hc_ac(
                    hc,
                    ac,
                )
            )

    if not actual:
        raise SystemExit(
            "No valid rows found."
        )

    print()
    print("=" * 78)
    print("EFW FORMULA VALIDATION")
    print("=" * 78)

    for name, prediction in (
        predictions.items()
    ):
        m = metrics(
            actual,
            prediction,
        )

        print()
        print(name)
        print("-" * 78)

        for k, v in m.items():
            if isinstance(v, float):
                print(
                    f"{k:20s}: {v:.3f}"
                )
            else:
                print(
                    f"{k:20s}: {v}"
                )


if __name__ == "__main__":
    main()
