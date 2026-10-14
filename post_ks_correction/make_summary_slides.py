"""Three lab-meeting slides: how the benchmark was built, what each Kilosort
variant scored, and why merging fails.

NUMBERS AND CONVENTION. Every figure here is recomputed from
outputs/v2_scores.csv and matches the Shamir Sorting Audit exactly, because it
uses the audit's convention: a plain pooled mean over all 52 placements and
ALL FOUR tiers. (An earlier draft quoted paired-Wilcoxon deltas over the three
measurable tiers and a graph-only footprint hook that is not in the audit;
those are different quantities and did not reconcile.) Checks against the
audit: stock easy 88/76, collision 83/59, pair 72/49; footprint x3 +1.8/+2.2;
footprint x1 -2.3/+5.4.
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
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
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


def bullets(slide, x, y, w, title, items, size=10.5):
    f = tb(slide, x, y, w, 0.26)
    run(f.paragraphs[0], title, 12, True, TEAL)
    g = tb(slide, x, y + 0.34, w, 1.0)
    for i, (lead, rest) in enumerate(items):
        p = g.paragraphs[0] if i == 0 else g.add_paragraph()
        if i:
            p.space_before = Pt(6)
        run(p, lead + "  ", size, True, INK)
        run(p, rest, size, False, MUTED)
    return g


# ============================================================== SLIDE 1
s1 = prs.slides.add_slide(BLANK)
header(s1, "How the test data was made",
       "Ground truth \u00b7 rat Shamir, striatal Neuropixels")

band(s1, 0.62, 1.46, 12.1, 0.52, TINT)
f = tb(s1, 0.88, 1.60, 11.6, 0.3)
p = f.paragraphs[0]
run(p, "On a real recording nobody knows which neuron a spike came from.  ",
    12.5, True, INK)
run(p, "So we built recordings where we do: real voltage with its real "
       "neurons still in it, plus a few donor neurons whose every spike time "
       "we chose ourselves.", 12.5, False, INK)

bullets(s1, 0.62, 2.20, 5.75, "1 \u00b7 Choosing the donor neurons", [
    ("Hand-reviewed.",
     "Gil reviewed every candidate in the review tool; the 33 labelled "
     "good became the donor pool."),
    ("Then filtered.",
     "low contamination (clean refractory period), and a compact spatial "
     "footprint."),
    ("Pairs checked separately.",
     "a cross-correlogram refractory test confirms the two donors are really "
     "different cells, not one cell Kilosort had split."),
    ("Real snippets, not templates.",
     "we inject the neuron's own recorded spikes. A copied template is "
     "identical every time, which flatters any sorter \u2014 real spikes vary "
     "in shape and amplitude, and that variation is what makes sorting hard."),
])

bullets(s1, 6.98, 2.20, 5.74, "2 \u00b7 Choosing where to put them", [
    ("Sites ranked by their loudest resident.",
     "every probe site scored by the amplitude of the biggest neuron already "
     "living there \u2014 23 to 467 \u00b5V across this probe, a 20\u00d7 "
     "contrast to build difficulty from."),
    ("Spikes dodge the residents.",
     "injection times keep a 75-sample guard from a resident neuron's own "
     "spikes \u2014 except on the collision tier, where the overlap is the "
     "point."),
    ("Each unit rotates through tiers.",
     "a donor appears at a different difficulty in each recording, so a tier "
     "difference is measured within a neuron rather than between neurons."),
])

tiers = [
    ("easy", "3", "quiet site, 23\u201370 \u00b5V neighbours",
     "nothing \u2014 this is the control"),
    ("collision", "3", "quiet site",
     "each spike placed 4\u201325 samples (0.13\u20130.83 ms) from a real "
     "resident's spike, so the two waveforms physically overlap"),
    ("pair", "2 pairs", "quiet sites, 3 channels apart",
     "two donors put side by side \u2014 they must be told apart from each "
     "other. Remote in the real recording (181\u2013242 \u00b5m, similarity "
     "0.000), so they cannot be one split cell"),
    ("hard", "3", "loud site, 223\u2013466 \u00b5V neighbours",
     "a large neighbour dominates the channel"),
]
top = 4.68
f = tb(s1, 0.62, top - 0.30, 7.0, 0.26)
run(f.paragraphs[0], "3 \u00b7 The four conditions", 12, True, TEAL)
f = tb(s1, 4.30, top - 0.28, 8.42, 0.24, PP_ALIGN.RIGHT)
run(f.paragraphs[0],
    "13 placements per recording \u00b7 4 recordings from different windows "
    "of the session \u00b7 ~290 spikes each", 10, False, MUTED)

rule(s1, 0.62, top, 12.1, INK, 0.016)
for lbl, x, w in (("TIER", 0.86, 1.3), ("PER REC.", 2.16, 0.9),
                  ("WHERE IT GOES", 3.20, 2.9),
                  ("WHAT MAKES IT HARD", 6.25, 6.3)):
    hf = tb(s1, x, top + 0.08, w, 0.22)
    run(hf.paragraphs[0], lbl, 9, True, MUTED)

y = top + 0.36
for i, (lbl, per, where, why) in enumerate(tiers):
    h = 0.48
    if lbl == "hard":
        band(s1, 0.62, y - 0.04, 12.1, h, RGBColor(0xF7, 0xF2, 0xF2))
    f = tb(s1, 0.86, y + 0.02, 1.3, 0.3)
    run(f.paragraphs[0], lbl, 12, True, INK if lbl != "hard" else RED)
    f = tb(s1, 2.16, y + 0.03, 0.9, 0.3)
    run(f.paragraphs[0], per, 10.5, False, INK)
    f = tb(s1, 3.20, y + 0.03, 2.95, 0.4)
    run(f.paragraphs[0], where, 10, False, INK)
    f = tb(s1, 6.25, y + 0.01, 6.3, 0.46)
    run(f.paragraphs[0], why, 10, False, MUTED)
    if i < 3:
        rule(s1, 0.86, y + h - 0.04, 11.62)
    y += h

f = tb(s1, 0.62, y + 0.08, 12.1, 0.3)
p = f.paragraphs[0]
run(p, "52 placements \u00b7 15,086 ground-truth spikes.  ", 10.5, True, INK)
run(p, "Stock Kilosort is also run twice as a configuration \u2014 the noise "
       "floor, and how we know the hard tier is unreliable.", 10.5, False,
    MUTED)

# ============================================================== SLIDE 2
s2 = prs.slides.add_slide(BLANK)
header(s2, "What each variant scored",
       "Results \u00b7 52 placements, pooled over 4 replicates")

# -- stock by tier
f = tb(s2, 0.62, 1.44, 6.0, 0.26)
run(f.paragraphs[0], "Stock Kilosort4, by condition", 12.5, True, TEAL)
f = tb(s2, 5.0, 1.47, 7.72, 0.24, PP_ALIGN.RIGHT)
run(f.paragraphs[0],
    "recall holds up everywhere \u00b7 precision collapses where neurons sit "
    "close together", 10, False, MUTED)

cells = [("easy", "88", "76"), ("collision", "83", "59"),
         ("pair", "72", "49"), ("hard", "\u2014", "\u2014")]
x = 0.62
for lbl, rc, pc in cells:
    is_hard = lbl == "hard"
    band(s2, x, 1.78, 2.86, 0.86, RGBColor(0xF7, 0xF2, 0xF2) if is_hard
         else BAND)
    f = tb(s2, x + 0.22, 1.88, 1.3, 0.26)
    run(f.paragraphs[0], lbl, 11.5, True, RED if is_hard else INK)
    if is_hard:
        f = tb(s2, x + 0.22, 2.14, 2.5, 0.4)
        run(f.paragraphs[0], "not reproducible \u2014 never quote it",
            9.5, False, RED)
    else:
        for val, name, dx in ((rc, "recall", 0.22), (pc, "precision", 1.45)):
            f = tb(s2, x + dx, 2.12, 1.2, 0.36)
            run(f.paragraphs[0], val, 21, True, INK)
            run(f.paragraphs[0], "%", 12, True, MUTED)
            g = tb(s2, x + dx, 2.44, 1.2, 0.2)
            run(g.paragraphs[0], name, 9, False, MUTED)
    x += 3.08

# -- variant table
rows = [
    ("Stock Kilosort4", "the baseline", "79.4", "60.0", "", None, True),
    ("Coarse + fine timing",
     "slide the snippet \u00b16 whole samples against the average spike "
     "shape, then correct the leftover sub-sample error from wavelet phase",
     "77.1", "58.4", "no gain on either", MUTED, False),
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

top = 2.92
rule(s2, 0.62, top, 12.1, INK, 0.016)
for lbl, x, w, al in (("ALGORITHM", 0.86, 2.1, PP_ALIGN.LEFT),
                      ("WHAT IT CHANGES", 3.00, 4.70, PP_ALIGN.LEFT),
                      ("RECALL", 7.86, 1.1, PP_ALIGN.RIGHT),
                      ("PRECISION", 9.06, 1.1, PP_ALIGN.RIGHT),
                      ("VS STOCK", 10.36, 2.36, PP_ALIGN.RIGHT)):
    hf = tb(s2, x, top + 0.08, w, 0.22, al)
    run(hf.paragraphs[0], lbl, 9, True, MUTED)

y = top + 0.38
for lbl, desc, rc, pc, note, ncol, is_base in rows:
    h = 0.76
    if is_base:
        band(s2, 0.62, y - 0.05, 12.1, h, RGBColor(0xEC, 0xEF, 0xF2))
    f = tb(s2, 0.86, y + 0.06, 2.1, 0.3)
    run(f.paragraphs[0], lbl, 12.5, True, INK)
    f = tb(s2, 3.00, y + 0.04, 4.70, 0.66)
    run(f.paragraphs[0], desc, 9.5, False, MUTED)
    for val, dx in ((rc, 7.86), (pc, 9.06)):
        f = tb(s2, dx, y + 0.06, 1.1, 0.34, PP_ALIGN.RIGHT)
        run(f.paragraphs[0], val, 17, True, INK)
        run(f.paragraphs[0], "%", 10.5, True, MUTED)
    f = tb(s2, 10.36, y + 0.10, 2.36, 0.5, PP_ALIGN.RIGHT)
    for j, part in enumerate(note.split("\n")):
        p = f.paragraphs[0] if j == 0 else f.add_paragraph()
        p.alignment = PP_ALIGN.RIGHT
        run(p, part, 9.5, True, ncol if ncol else MUTED)
    if not is_base:
        rule(s2, 0.86, y + h - 0.05, 11.62)
    y += h

band(s2, 0.62, y + 0.06, 12.1, 0.52, TINT)
f = tb(s2, 0.88, y + 0.18, 11.6, 0.3)
p = f.paragraphs[0]
run(p, "Why footprint appears twice:  ", 11, True, TEAL)
run(p, "it is one idea at two strengths, and the weight was the experiment. "
       "\u00d71 buys purity, \u00d73 buys completeness \u2014 neither wins "
       "both, so the choice depends on what the analysis needs.",
    11, False, INK)

f = tb(s2, 0.62, y + 0.68, 12.1, 0.5)
p = f.paragraphs[0]
run(p, "Pooled over all four tiers, as in the Shamir Sorting Audit. Noise "
       "floor: stock run twice differs by 0.9 recall / 1.6 precision, all of "
       "it on the hard tier.", 9.5, False, MUTED)
p2 = f.add_paragraph()
p2.space_before = Pt(5)
run(p2, "Scores are for the one best-matching cluster per injected unit "
        "\u2014 13 of the ~200 clusters a run produces.  ", 9.5, True, INK)
run(p2, "The rest are the recording's own neurons, which have no ground "
        "truth and are never scored.", 9.5, False, MUTED)

# ============================================================== SLIDE 3
s3 = prs.slides.add_slide(BLANK)
header(s3, "Why merging the pieces back together does not help",
       "The obvious fix")

f = tb(s3, 0.62, 1.50, 12.1, 0.4)
run(f.paragraphs[0],
    "We merged while being told the right answer \u2014 the best any merge "
    "tool could ever do.", 16, True, INK)

stats = [
    ("0 of 16", "algorithm \u00d7 tier cells where merging won",
     "it lost in every one"),
    ("+5 / \u221212", "typical recall gained / precision lost",
     "best case anywhere: +3.2 / \u22123.6"),
    ("80.6%", "of what comes in is another neuron",
     "a real spike, not noise \u2014 chance is 3.9%"),
]
x = 0.62
for big, lab, sub in stats:
    band(s3, x, 2.12, 3.95, 1.52)
    f = tb(s3, x + 0.3, 2.28, 3.4, 0.5)
    run(f.paragraphs[0], big, 34, True, TEAL)
    f = tb(s3, x + 0.3, 2.86, 3.4, 0.3)
    run(f.paragraphs[0], lab, 11.5, True, INK)
    f = tb(s3, x + 0.3, 3.18, 3.4, 0.3)
    run(f.paragraphs[0], sub, 10, False, MUTED)
    x += 4.19

lines = [
    ("There was little to merge in the first place.",
     "67\u201377% of units already sit in one cluster."),
    ("Merging imports a neighbour's spike train,",
     "so the cluster stops being a single cell."),
    ("A perfect merge tool makes every variant identical,",
     "all converging to about 78% recall / 46% precision."),
]
y = 3.98
for head, tail in lines:
    f = tb(s3, 0.88, y, 11.6, 0.34)
    p = f.paragraphs[0]
    run(p, head + " ", 14, True, INK)
    run(p, tail, 14, False, MUTED)
    y += 0.52

band(s3, 0.62, y + 0.18, 12.1, 1.02, TINT)
f = tb(s3, 0.95, y + 0.36, 11.4, 0.7)
p = f.paragraphs[0]
run(p, "So the failure is not detection and not splitting.", 15, True, TEAL)
p2 = f.add_paragraph()
p2.space_before = Pt(6)
run(p2, "94\u201396% of injected spikes do land in some cluster. The "
        "problem is which cluster each spike goes to \u2014 so the lead is "
        "the opposite operation: splitting Kilosort's mixed clusters.",
    13, False, INK)

out = r"D:\Gil\spike_sorting_agent\outputs\kilosort_variants_summary.pptx"
prs.save(out)
print("saved", out)
