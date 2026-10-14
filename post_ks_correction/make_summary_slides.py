"""Three lab-meeting slides: how the benchmark was built, what each variant
scored, and why merging fails.

NUMBERS AND CONVENTION. Every figure is recomputed from outputs/v2_scores.csv
with the Shamir Sorting Audit's convention: a plain pooled mean over all 52
placements and ALL FOUR tiers. Check values against the audit: stock easy
88/76, collision 83/59, pair 72/49; footprint x3 +1.8/+2.2; footprint x1
-2.3/+5.4. Do NOT substitute paired-Wilcoxon deltas or the graph-only
footprint hook from 5aw -- neither is in the audit and they will not
reconcile.

The precision-matrix row is the POST-HOC pipeline (template convergence ->
AR(4) noise whitening -> matched filter), not a Kilosort patch. Audit pooled
figures: 11,027 accepted, 259 real (2.3%), +1.7 recall / -4.3 precision on top
of stock, i.e. 81.1 / 55.7.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

INK = RGBColor(0x1A, 0x1F, 0x2E)
MUTED = RGBColor(0x6B, 0x72, 0x80)
TEAL = RGBColor(0x0F, 0x6E, 0x6E)
RED = RGBColor(0xB3, 0x26, 0x1E)
GREEN = RGBColor(0x1B, 0x6B, 0x3A)
RULE = RGBColor(0xD7, 0xDB, 0xE0)
BAND = RGBColor(0xF4, 0xF6, 0xF8)
TINT = RGBColor(0xE8, 0xF1, 0xF1)
PINK = RGBColor(0xF7, 0xF2, 0xF2)
BODY = "Segoe UI"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    f = box.text_frame
    f.word_wrap = True
    f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
    f.paragraphs[0].alignment = align
    return f


def run(p, text, size=12, bold=False, color=INK):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = BODY
    return r


def rule(slide, x, y, w, color=RULE, h=0.013):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def band(slide, x, y, w, h, color=BAND):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                               Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.035
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def header(slide, title, kicker):
    f = tb(slide, 0.62, 0.40, 12.1, 0.3)
    r = run(f.paragraphs[0], kicker.upper(), 10.5, True, TEAL)
    r.font._rPr.set("spc", "120")
    g = tb(slide, 0.62, 0.70, 12.1, 0.5)
    run(g.paragraphs[0], title, 26, True, INK)
    rule(slide, 0.62, 1.28, 12.1)


# ============================================================== SLIDE 1
s1 = prs.slides.add_slide(BLANK)
header(s1, "How the test data was made",
       "Ground truth \u00b7 rat Shamir, striatal Neuropixels")

f = tb(s1, 0.62, 1.52, 12.1, 0.35)
p = f.paragraphs[0]
run(p, "Real voltage, real neurons \u2014 plus donor neurons whose every "
       "spike time we chose.", 17, True, INK)

for x, w, title, text in [
    (0.62, 3.85, "The donors",
     "33 units Gil labelled good, filtered for clean refractory and compact "
     "footprint. We inject their own recorded spikes \u2014 not copied "
     "templates, which flatter any sorter."),
    (4.74, 3.85, "Where they go",
     "Every probe site ranked by the amplitude of the biggest neuron already "
     "living there: 23 to 467 \u00b5V, a 20\u00d7 span to build difficulty "
     "from."),
    (8.86, 3.85, "The scale",
     "13 placements \u00d7 4 recordings from different windows of the "
     "session. 52 placements, 15,086 ground-truth spikes, ~290 per unit."),
]:
    band(s1, x, 2.00, w, 1.34)
    f = tb(s1, x + 0.28, 2.16, w - 0.56, 0.26)
    run(f.paragraphs[0], title, 12.5, True, TEAL)
    f = tb(s1, x + 0.28, 2.48, w - 0.56, 0.78)
    run(f.paragraphs[0], text, 11, False, INK)

tiers = [
    ("easy", "quiet site, 23\u201370 \u00b5V neighbours",
     "nothing \u2014 the control", False),
    ("collision", "quiet site",
     "each spike overlaps a resident's spike, 0.13\u20130.83 ms apart", False),
    ("pair", "quiet sites, 3 channels apart",
     "two donors side by side \u2014 must be told apart from each other",
     False),
    ("hard", "loud site, 223\u2013466 \u00b5V neighbours",
     "a large neighbour dominates the channel", True),
]
top = 3.86
f = tb(s1, 0.62, top - 0.34, 7.0, 0.3)
run(f.paragraphs[0], "The four conditions", 15, True, TEAL)

rule(s1, 0.62, top, 12.1, INK, 0.016)
for lbl, x, w in (("TIER", 0.86, 1.6), ("WHERE IT GOES", 2.65, 3.6),
                  ("WHAT MAKES IT HARD", 6.50, 6.1)):
    hf = tb(s1, x, top + 0.09, w, 0.22)
    run(hf.paragraphs[0], lbl, 9, True, MUTED)

y = top + 0.40
for i, (lbl, where, why, is_hard) in enumerate(tiers):
    h = 0.62
    if is_hard:
        band(s1, 0.62, y - 0.05, 12.1, h, PINK)
    f = tb(s1, 0.86, y + 0.10, 1.7, 0.3)
    run(f.paragraphs[0], lbl, 15, True, RED if is_hard else INK)
    f = tb(s1, 2.65, y + 0.14, 3.6, 0.4)
    run(f.paragraphs[0], where, 11.5, False, INK)
    f = tb(s1, 6.50, y + 0.14, 6.1, 0.4)
    run(f.paragraphs[0], why, 11.5, False, MUTED)
    if i < 3:
        rule(s1, 0.86, y + h - 0.05, 11.62)
    y += h

f = tb(s1, 0.62, y + 0.18, 12.1, 0.3)
p = f.paragraphs[0]
run(p, "Each donor rotates through tiers across the four recordings, ",
    11, False, MUTED)
run(p, "so a tier difference is measured within a neuron, not between "
       "neurons.", 11, True, INK)

# ============================================================== SLIDE 2
s2 = prs.slides.add_slide(BLANK)
header(s2, "What each variant scored",
       "Results \u00b7 52 placements, pooled over 4 replicates")

f = tb(s2, 0.62, 1.42, 6.0, 0.26)
run(f.paragraphs[0], "Stock Kilosort4, by condition", 12.5, True, TEAL)
f = tb(s2, 5.0, 1.45, 7.72, 0.24, PP_ALIGN.RIGHT)
run(f.paragraphs[0],
    "recall holds up everywhere \u00b7 precision collapses where neurons sit "
    "close together", 10, False, MUTED)

cells = [("easy", "88", "76"), ("collision", "83", "59"),
         ("pair", "72", "49"), ("hard", "\u2014", "\u2014")]
x = 0.62
for lbl, rc, pc in cells:
    is_hard = lbl == "hard"
    band(s2, x, 1.74, 2.86, 0.84, PINK if is_hard else BAND)
    f = tb(s2, x + 0.22, 1.83, 1.6, 0.26)
    run(f.paragraphs[0], lbl, 11.5, True, RED if is_hard else INK)
    if is_hard:
        f = tb(s2, x + 0.22, 2.09, 2.5, 0.4)
        run(f.paragraphs[0], "not reproducible \u2014 never quote it",
            9.5, False, RED)
    else:
        for val, name, dx in ((rc, "recall", 0.22), (pc, "precision", 1.45)):
            f = tb(s2, x + dx, 2.07, 1.2, 0.36)
            run(f.paragraphs[0], val, 21, True, INK)
            run(f.paragraphs[0], "%", 12, True, MUTED)
            g = tb(s2, x + dx, 2.39, 1.2, 0.2)
            run(g.paragraphs[0], name, 9, False, MUTED)
    x += 3.08

rows = [
    ("Stock Kilosort4", "the baseline",
     "79.4", "60.0", "", None, True),
    ("Coarse + fine timing",
     "slide the snippet \u00b16 whole samples against the average spike "
     "shape, then correct the leftover sub-sample error from wavelet phase",
     "77.1", "58.4", "no gain on either", MUTED, False),
    ("Noise precision matrix",
     "converge the template, whiten against the AR(4) noise covariance, then "
     "matched-filter the session for missed spikes   \u00b7   runs AFTER "
     "Kilosort, not inside it",
     "81.1", "55.7", "of 11,027 spikes recovered,\nonly 2.3% were the "
     "unit's", RED, False),
    ("Footprint \u00d71",
     "add the spike's spatial footprint \u2014 its pattern of amplitude "
     "across nearby channels \u2014 to what the clustering sees, weighted "
     "equal to the waveform",
     "77.1", "65.4", "precision +5.4  (p = 0.010)", GREEN, False),
    ("Footprint \u00d73",
     "the same feature weighted three times as strongly, so footprint "
     "dominates the waveform",
     "81.2", "62.1", "recall +1.8, precision +2.2\npair recall 72\u219280 "
     "(p = 0.038)", GREEN, False),
]

top = 2.78
rule(s2, 0.62, top, 12.1, INK, 0.016)
for lbl, x, w, al in (("ALGORITHM", 0.86, 2.1, PP_ALIGN.LEFT),
                      ("WHAT IT CHANGES", 3.00, 4.70, PP_ALIGN.LEFT),
                      ("RECALL", 7.86, 1.1, PP_ALIGN.RIGHT),
                      ("PRECISION", 9.06, 1.1, PP_ALIGN.RIGHT),
                      ("VS STOCK", 10.36, 2.36, PP_ALIGN.RIGHT)):
    hf = tb(s2, x, top + 0.08, w, 0.22, al)
    run(hf.paragraphs[0], lbl, 9, True, MUTED)

y = top + 0.36
for lbl, desc, rc, pc, note, ncol, is_base in rows:
    h = 0.64
    if is_base:
        band(s2, 0.62, y - 0.05, 12.1, h, RGBColor(0xEC, 0xEF, 0xF2))
    f = tb(s2, 0.86, y + 0.04, 2.1, 0.3)
    run(f.paragraphs[0], lbl, 12.5, True, INK)
    f = tb(s2, 3.00, y + 0.02, 4.70, 0.64)
    run(f.paragraphs[0], desc, 9.5, False, MUTED)
    for val, dx in ((rc, 7.86), (pc, 9.06)):
        f = tb(s2, dx, y + 0.04, 1.1, 0.34, PP_ALIGN.RIGHT)
        run(f.paragraphs[0], val, 17, True, INK)
        run(f.paragraphs[0], "%", 10.5, True, MUTED)
    f = tb(s2, 10.36, y + 0.08, 2.36, 0.5, PP_ALIGN.RIGHT)
    for j, part in enumerate(note.split("\n")):
        p = f.paragraphs[0] if j == 0 else f.add_paragraph()
        p.alignment = PP_ALIGN.RIGHT
        run(p, part, 9.5, True, ncol if ncol else MUTED)
    if not is_base:
        rule(s2, 0.86, y + h - 0.05, 11.62)
    y += h

band(s2, 0.62, y + 0.04, 12.1, 0.48, TINT)
f = tb(s2, 0.88, y + 0.15, 11.6, 0.3)
p = f.paragraphs[0]
run(p, "Why footprint appears twice:  ", 11, True, TEAL)
run(p, "one idea at two strengths, and the weight was the experiment. "
       "\u00d71 buys purity, \u00d73 buys completeness \u2014 neither wins "
       "both.", 11, False, INK)

f = tb(s2, 0.62, y + 0.60, 12.1, 0.45)
p = f.paragraphs[0]
run(p, "Pooled over all four tiers, as in the Shamir Sorting Audit. Noise "
       "floor: stock run twice differs by 0.9 recall / 1.6 precision, all on "
       "the hard tier.", 9.5, False, MUTED)
p2 = f.add_paragraph()
p2.space_before = Pt(4)
run(p2, "Scores are for the one best-matching cluster per injected unit "
        "\u2014 13 of the ~200 a run produces.  ", 9.5, True, INK)
run(p2, "The rest are the recording's own neurons, which have no ground "
        "truth and are never scored.", 9.5, False, MUTED)

# ============================================================== SLIDE 3
s3 = prs.slides.add_slide(BLANK)
header(s3, "Why merging the pieces back together does not help",
       "The obvious fix")

f = tb(s3, 0.62, 1.70, 12.1, 0.45)
run(f.paragraphs[0],
    "We merged while being told the right answer.", 24, True, INK)

stats = [
    ("0 of 16", "conditions where merging won"),
    ("+5  /  \u221212", "recall gained / precision lost"),
    ("80.6%", "of what comes in is another neuron"),
]
x = 0.62
for big, lab in stats:
    band(s3, x, 2.72, 3.95, 1.56)
    f = tb(s3, x + 0.32, 2.98, 3.3, 0.6)
    run(f.paragraphs[0], big, 40, True, TEAL)
    f = tb(s3, x + 0.32, 3.70, 3.3, 0.4)
    run(f.paragraphs[0], lab, 12, True, INK)
    x += 4.19

band(s3, 0.62, 4.76, 12.1, 1.22, TINT)
f = tb(s3, 1.0, 5.06, 11.3, 0.7)
run(f.paragraphs[0],
    "You are not merging back the missing spikes. You are merging in a "
    "neighbour.", 21, True, TEAL)

f = tb(s3, 0.62, 6.30, 12.1, 0.3)
run(f.paragraphs[0],
    "A real spike, not noise \u2014 chance is 3.9%. Most units were barely "
    "split to begin with.", 12, False, MUTED)

out = r"D:\Gil\spike_sorting_agent\outputs\kilosort_variants_summary.pptx"
prs.save(out)
print("saved", out)
