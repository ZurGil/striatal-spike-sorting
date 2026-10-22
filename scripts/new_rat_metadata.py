# -*- coding: utf-8 -*-
"""Write a new rat's `rat_metadata.json`, validated by the pipeline's own loader.

    python scripts/new_rat_metadata.py --rat_id Newname_1 --sex M \
        --date_of_birth 2026-06-01 --implant_date 2026-10-20 \
        --probe_serial_number 19108309999 \
        --implant_ML_mm -2.45 --implant_AP_mm 1.2 --implant_depth_mm 7.5 \
        --hemisphere L

The sync pipeline needs one hand-written file per rat, at
`<rat_root>/rat_metadata.json`. A missing key is not caught until the pipeline runs,
which is after the recording, so this writes the file through
`sync_pipeline.rat_metadata.load_rat_metadata` -- the exact function the pipeline uses
-- and refuses to leave a file behind that would not load.

Four of the fields are not labels. They are arithmetic:

  implant_depth_mm   every unit's depth_mm = implant_depth_mm - (phy depth / 1000).
                     Positive, measured from the brain surface down to the probe tip.
  implant_angle_deg  a NON-ZERO value is only WARNED about, never corrected -- the
                     pipeline's depth formula assumes a straight-down implant, so a
                     tilted probe needs a cos(angle) correction applied downstream.
  implant_ML_mm      copied verbatim onto every unit row as ML_mm.
  implant_AP_mm      likewise as AP_mm.

So these have no defaults here on purpose. Inheriting the previous rat's coordinates
by accident would mislabel the recording site of every unit, quietly and for good.
`probe_type` and `surgeon` do default, since they are provenance and wrong is visible.

`probe_marker` records the fluorescent dye painted on the shank, so that months later
nobody has to guess which filter the histology needs. It is NOT one of the pipeline's
required fields -- `load_rat_metadata` checks only that the required keys are present
and ignores extra ones, and `process_session` embeds the whole file verbatim into
`session_meta.json["rat_metadata"]`. So it rides along at no risk.
"""
import argparse
import importlib.util
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAT_PARENT = os.path.join("Z:" + os.sep, "Gil")


def pipeline_loader():
    """`load_rat_metadata` from the sibling sync_pipeline checkout.

    Loaded straight from its file rather than imported as a package member: the module
    depends only on json and pathlib, and this way the validation does not need
    sync_pipeline to be importable as a package.
    """
    env = os.environ.get("SYNC_PIPELINE_DIR")
    cands = ([env] if env else []) + [os.path.dirname(ROOT)]
    for c in cands:
        path = os.path.join(c, "sync_pipeline", "rat_metadata.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_pipeline_rat_metadata", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, path
    raise SystemExit(
        "cannot find sync_pipeline/rat_metadata.py. It is a sibling checkout rather "
        "than an installed module, so it should sit next to this repo (expected "
        "%s). Set SYNC_PIPELINE_DIR to its parent directory if it lives elsewhere."
        % os.path.join(os.path.dirname(ROOT), "sync_pipeline", "rat_metadata.py"))


def probe_marker(a):
    """The dye painted on the shank, and what it takes to see it later.

    The identification fields are read off the vial. The spectral numbers are NOT:
    Thermo's page for V22885 describes the dye qualitatively (orange-red, spectrally
    similar to tetramethylrhodamine) and the lot certificate's 549 nm is an absorbance
    QC specification rather than a stated excitation peak. They are labelled nominal
    here rather than dressed up as measured, because the whole point of the field is
    that somebody trusts it months from now.
    """
    marker = {
        "dye": a.marker_dye,
        "product": a.marker_product,
        "manufacturer": "Invitrogen / Thermo Fisher Scientific "
                        "(Life Technologies Corporation, Eugene, Oregon)",
        "catalog_number": a.marker_catalog,
        "lot": a.marker_lot,
        "concentration": a.marker_concentration,
        "storage": "-25 to 30 C, protect from light",
        "imaging": {
            "fluorescence": "orange-red; the vendor describes it as spectrally "
                            "similar to tetramethylrhodamine",
            "excitation_nm_nominal": 549,
            "emission_nm_nominal": 565,
            "filter_set": "TRITC / Cy3 / rhodamine -- green-yellow excitation, "
                          "orange-red emission",
            "spectra_caveat": "the nm values are the usual figures quoted for DiI and "
                              "were NOT confirmed against the vendor's own spectrum. "
                              "Check the V22885 spectrum before relying on them. Note "
                              "that some TRITC emission filters start near 570 nm and "
                              "would clip a 565 nm peak -- one centred ~570-590 nm is "
                              "a safer choice.",
        },
    }
    if a.marker_photo:
        marker["label_photo"] = a.marker_photo
    return marker


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rat_id", required=True,
                   help="e.g. Newname_1. Also names the folder under the rat parent.")
    p.add_argument("--sex", required=True, choices=["M", "F"])
    p.add_argument("--date_of_birth", required=True, help="YYYY-MM-DD")
    p.add_argument("--implant_date", required=True, help="YYYY-MM-DD")
    p.add_argument("--probe_serial_number", required=True)
    p.add_argument("--implant_ML_mm", required=True, type=float,
                   help="negative = left of midline. Becomes units.ML_mm.")
    p.add_argument("--implant_AP_mm", required=True, type=float,
                   help="positive = anterior to bregma. Becomes units.AP_mm.")
    p.add_argument("--implant_depth_mm", required=True, type=float,
                   help="POSITIVE: brain surface down to the probe tip. units.depth_mm "
                        "is computed from this.")
    p.add_argument("--hemisphere", required=True, choices=["L", "R"])
    p.add_argument("--implant_angle_deg", type=float, default=0.0,
                   help="default 0 (straight down). Non-zero is warned about by the "
                        "pipeline but NOT corrected for.")
    p.add_argument("--probe_type", default="Neuropixels 1.0")
    p.add_argument("--surgeon", default="Gil")
    p.add_argument("--headstage_serial_number", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--marker_lot", default=None,
                   help="lot number of the dye on the shank. Defaults to the DiI "
                        "product below; pass --no_marker to leave the field out.")
    p.add_argument("--marker_dye", default="DiI")
    p.add_argument("--marker_product", default="Vybrant DiI cell-labeling solution")
    p.add_argument("--marker_catalog", default="V22885")
    p.add_argument("--marker_concentration", default="1 mM in solvent")
    p.add_argument("--marker_photo", default=None,
                   help="filename of a photo of the vial label, kept beside this file")
    p.add_argument("--no_marker", action="store_true",
                   help="omit probe_marker entirely (no dye, or not recorded)")
    p.add_argument("--rat_parent", default=DEFAULT_RAT_PARENT,
                   help="where rat folders live (default %s)" % DEFAULT_RAT_PARENT)
    p.add_argument("--out", default=None,
                   help="full path to write. Default <rat_parent>/<rat_id>/"
                        "rat_metadata.json")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing file (it is NOT overwritten otherwise)")
    p.add_argument("--dry_run", action="store_true",
                   help="print the file and validate it, but write nothing")
    a = p.parse_args()

    loader, loader_path = pipeline_loader()

    data_root = os.path.join(a.rat_parent, a.rat_id)
    out = a.out or os.path.join(data_root, "rat_metadata.json")

    notes = a.notes or (
        "implant_depth_mm is a positive value = distance in mm from brain surface "
        "down to the probe tip."
        + ("" if a.headstage_serial_number else " headstage_serial_number pending."))

    # key order follows the existing Shamir_1 and Begin_1 files, so the three are
    # diffable against each other
    meta = {
        "rat_id": a.rat_id,
        "sex": a.sex,
        "date_of_birth": a.date_of_birth,
        "implant_date": a.implant_date,
        "probe_type": a.probe_type,
        "probe_serial_number": a.probe_serial_number,
        "headstage_serial_number": a.headstage_serial_number,
        "implant_ML_mm": a.implant_ML_mm,
        "implant_AP_mm": a.implant_AP_mm,
        "implant_depth_mm": a.implant_depth_mm,
        "implant_angle_deg": a.implant_angle_deg,
        "hemisphere": a.hemisphere,
        "surgeon": a.surgeon,
        "data_root": data_root,
        "notes": notes,
    }

    if not a.no_marker:
        meta["probe_marker"] = probe_marker(a)

    text = json.dumps(meta, indent=2) + "\n"
    print(text)

    # sanity, beyond the pipeline's own which only checks that keys exist
    warn = []
    if a.implant_depth_mm <= 0:
        warn.append("implant_depth_mm is %s. It must be POSITIVE -- it is measured "
                    "from the brain surface DOWN to the tip, and a negative value "
                    "makes every unit depth wrong by twice the implant depth."
                    % a.implant_depth_mm)
    if a.hemisphere == "L" and a.implant_ML_mm > 0:
        warn.append("hemisphere is L but implant_ML_mm is positive (%s). Negative is "
                    "left of midline." % a.implant_ML_mm)
    if a.hemisphere == "R" and a.implant_ML_mm < 0:
        warn.append("hemisphere is R but implant_ML_mm is negative (%s)."
                    % a.implant_ML_mm)
    if a.implant_angle_deg != 0:
        warn.append("implant_angle_deg is %s. The pipeline's depth formula assumes a "
                    "straight-down implant and applies NO cos(angle) correction -- it "
                    "only logs a warning. Unit depths for this rat will need "
                    "correcting downstream." % a.implant_angle_deg)
    if a.date_of_birth >= a.implant_date:
        warn.append("date_of_birth (%s) is not before implant_date (%s)."
                    % (a.date_of_birth, a.implant_date))
    for w in warn:
        print("WARNING: %s" % w)
    if warn:
        print("")

    if os.path.exists(out) and not a.force and not a.dry_run:
        raise SystemExit("%s already exists. Pass --force to overwrite it, after "
                        "looking at what is there." % out)

    if a.dry_run:
        # validate via a temporary file so the pipeline's own loader is what judges it
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            io.open(tmp, "w", encoding="utf-8").write(text)
            loader.load_rat_metadata(tmp)
        finally:
            os.unlink(tmp)
        print("dry run: valid per %s, nothing written" % loader_path)
        return

    d = os.path.dirname(out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
        print("created %s" % d)
    io.open(out, "w", encoding="utf-8").write(text)

    # read it back with the pipeline's loader; a file that would crash the pipeline
    # does not get to survive this script
    try:
        back = loader.load_rat_metadata(out)
    except Exception as e:
        os.unlink(out)
        raise SystemExit("the file did not load with %s, so it was removed: %s"
                         % (loader_path, e))
    if back != meta:
        raise SystemExit("what was read back differs from what was written -- stop and "
                         "look at %s" % out)
    print("wrote %s" % out)
    print("validated with %s (%d required fields present)"
          % (loader_path, len(loader.REQUIRED_FIELDS)))
    print("")
    print("The pipeline finds this automatically as <rat_root>/rat_metadata.json, so")
    print("  python -m sync_pipeline.process_session --rec_folder \"%s\\<session>.rec\""
          % data_root)
    print("needs no extra argument for it.")


if __name__ == "__main__":
    main()
