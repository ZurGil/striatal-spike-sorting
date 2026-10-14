"""Two lab-meeting slides: the Kilosort variant benchmark, and why merging
fails. Every number is traceable to RESEARCH_LOG.md sections 5ak/5al/5aw/5ba.
All figures are pooled over 4 replicates (52 placements) -- never one
replicate, and recall is never shown without precision.
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
    s.adjustments[0] = 0.04
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def header(slide, title, kicker):
    f = tb(slide, 0.62, 0.42, 12.1, 0.3)
    r = run(f.paragraphs[0], kicker.upper(), 10.5, True, TEAL)
    r.font._rPr.set("spc", "120")
    g = tb(slide, 0.62, 0.72, 12.1, 0.5)
    run(g.paragraphs[0], title, 26, True, INK)
    rule(slide, 0.62, 1.32, 12.1)


# ----------------------------------------------------------------- SLIDE 1
s1 = prs.slides.add_slide(BLANK)
header(s1, "Kilosort4 variants on a hybrid ground-truth benchmark",
       "Spike sorting \u00b7 what we tried and what it bought")

band(s1, 0.62, 1.52, 12.1, 1.02)
f = tb(s1, 0.86, 1.70, 5.4, 0.75)
p = f.paragraphs[0]
run(p, "Dataset   ", 10.5, True, TEAL)
run(p, "Shamir striatal Neuropixels (20260916_110311). 33 hand-reviewed real "
       "units, their own recorded snippets re-injected into real voltage at "
       "known times.", 10.5, False, INK)
p2 = f.add_paragraph()
p2.space_before = Pt(4)
run(p2, "52 placements \u00d7 4 replicates (120 s windows), 15,086 truth "
        "spikes.", 10.5, False, MUTED)

f = tb(s1, 6.70, 1.70, 5.85, 0.75)
p = f.paragraphs[0]
run(p, "Conditions   ", 10.5, True, TEAL)
run(p, "set by the loudest resident neuron on the destination channel",
    10.5, False, MUTED)
for lbl, desc in [
        ("easy", "23\u201370 \u00b5V neighbour"),
        ("collision", "quiet site, spike 4\u201325 samples from a resident's"),
        ("pair", "two donors 181\u2013242 \u00b5m apart, similarity 0.000"),
        ("hard", "223\u2013466 \u00b5V \u2014 NOT measurable, excluded")]:
    pp = f.add_paragraph()
    pp.space_before = Pt(1)
    run(pp, lbl + "   ", 9.5, True, INK)
    run(pp, desc, 9.5, False, MUTED)

rows = [
    ("Stock Kilosort4", "baseline", ".794", ".600", None, None),
    ("+ Wavelet sub-sample alignment",
     "Morlet-wavelet phase places each spike to a fraction of a sample, so a "
     "template is not an average of smeared spikes",
     "n.s.", "n.s.", MUTED, MUTED),
    ("+ AR(4) temporal precision matrix",
     "Kilosort whitens channel\u00d7channel only; this models noise "
     "correlation in TIME   (tested post-hoc, not inside KS)",
     "+1.7", "\u22124.3", MUTED, RED),
    ("+ Amplitude normalisation",
     "divide out each spike's own amplitude before clustering",
     "\u22123.4", "n.s.", RED, MUTED),
    ("+ Spatial footprint \u2192 graph, w3",
     "the per-channel amplitude pattern added as extra clustering dimensions, "
     "inside Kilosort's kNN graph",
     "+4.8", "\u22123.3", GREEN, MUTED),
    ("+ Spatial footprint \u2192 graph+seed, w1",
     "same feature at lower weight, also shown to the kmeans++ seeding",
     "n.s.", "+6.6", MUTED, GREEN),
]

top = 2.78
rule(s1, 0.62, top, 12.1, INK, 0.016)
hf = tb(s1, 0.86, top + 0.09, 3.3, 0.22)
run(hf.paragraphs[0], "VARIANT", 9, True, MUTED)
hf = tb(s1, 4.35, top + 0.09, 5.9, 0.22)
run(hf.paragraphs[0], "WHAT IT ADDS", 9, True, MUTED)
for lbl, x in (("RECALL", 10.42), ("PRECISION", 11.52)):
    hf = tb(s1, x, top + 0.09, 1.05, 0.22, PP_ALIGN.RIGHT)
    run(hf.paragraphs[0], lbl, 9, True, MUTED)

y = top + 0.40
for i, (lbl, desc, rc, pc, rcol, pcol) in enumerate(rows):
    h = 0.52
    if i == 0:
        band(s1, 0.62, y - 0.05, 12.1, h, RGBColor(0xEC, 0xEF, 0xF2))
    f = tb(s1, 0.86, y + 0.02, 3.35, 0.5)
    run(f.paragraphs[0], lbl, 11.5, True, INK)
    f = tb(s1, 4.35, y + 0.01, 5.85, 0.54)
    run(f.paragraphs[0], desc, 9.5, False, MUTED)
    for val, col, x in ((rc, rcol, 10.42), (pc, pcol, 11.52)):
        f = tb(s1, x, y + 0.02, 1.05, 0.3, PP_ALIGN.RIGHT)
        run(f.paragraphs[0], val, 13 if i == 0 else 12.5, True,
            col if col else INK)
    if i < len(rows) - 1:
        rule(s1, 0.86, y + h - 0.05, 11.62)
    y += h

f = tb(s1, 0.62, y + 0.10, 12.1, 0.9)
p = f.paragraphs[0]
run(p, "Points vs stock, paired Wilcoxon on identical placements, "
       "easy+collision+pair only.  ", 9.5, False, MUTED)
run(p, "Significant: footprint-graph-w3 recall p=0.001 (better in all four "
       "replicates) \u00b7 footprint-w1 precision p=0.009 \u00b7 amplitude "
       "normalisation p=0.012.  ", 9.5, True, INK)
run(p, "n.s. = not distinguishable from stock. Hard tier excluded \u2014 two "
       "identical runs disagree by up to 0.444 recall.", 9.5, False, MUTED)
p2 = f.add_paragraph()
p2.space_before = Pt(8)
run(p2, "These scores are for the single best-matching cluster per injected "
        "unit \u2014 13 clusters of the ~238 a run produces.  ", 9.5, True,
    INK)
run(p2, "The other ~225 are the recording's own resident neurons, which have "
        "no ground truth and are never scored; injected units also sit in "
        "1.27\u20131.38 substantial clusters on average, so a second fragment "
        "of the same unit is left out of these numbers too.", 9.5, False,
    MUTED)

# ----------------------------------------------------------------- SLIDE 2
s2 = prs.slides.add_slide(BLANK)
header(s2, "Why merging the fragments back together does not help",
       "The obvious fix, tested at the oracle level")

f = tb(s2, 0.62, 1.60, 12.1, 0.45)
run(f.paragraphs[0],
    "We merged while knowing the true fragments \u2014 the best any merge "
    "tool could ever do. It still trades down everywhere.", 15, True, INK)

items = [
    ("There is barely anything to merge.",
     "67\u201377% of injected units already sit in ONE substantial cluster; "
     "the mean is 1.27\u20131.38 fragments and the maximum anywhere is 3. "
     "There is no three-way split waiting to be rejoined."),
    ("Combining loses in all 16 algorithm \u00d7 tier cells.",
     "+3 to +7 recall points bought for \u22124 to \u221222 precision. The "
     "best case anywhere, on the easiest tier, is +3.2 recall for "
     "\u22123.6 precision."),
    ("What comes in with the extra spikes is not noise \u2014 it is other "
     "neurons.",
     "80.6% of what counts against precision is a neighbouring resident "
     "neuron's REAL spike, against a measured chance rate of 3.9%. Merging a "
     "fragment imports a neighbour's spike train, so the cluster stops being "
     "one cell."),
    ("A perfect merge tool makes every variant the same.",
     "All configurations converge to \u2248 .78 recall / .46 precision once "
     "merged, so merging cannot separate the algorithms \u2014 and it erases "
     "the footprint gain that is the project's one real win."),
]

y = 2.18
for i, (head, bodytext) in enumerate(items):
    band(s2, 0.62, y, 12.1, 1.00, BAND if i % 2 == 0 else WHITE)
    nf = tb(s2, 0.88, y + 0.16, 0.4, 0.4)
    run(nf.paragraphs[0], str(i + 1), 17, True, TEAL)
    f = tb(s2, 1.36, y + 0.14, 11.1, 0.82)
    run(f.paragraphs[0], head, 12.5, True, INK)
    p2 = f.add_paragraph()
    p2.space_before = Pt(3)
    run(p2, bodytext, 10.5, False, MUTED)
    y += 1.07

band(s2, 0.62, y + 0.08, 12.1, 0.66, RGBColor(0xE8, 0xF1, 0xF1))
f = tb(s2, 0.88, y + 0.22, 11.6, 0.42)
p = f.paragraphs[0]
run(p, "So the loss is not detection and not splitting.  ", 11.5, True, TEAL)
run(p, "94\u201396% of injected spikes do land in some cluster \u2014 the "
       "failure is which cluster each spike is assigned to. The live lead is "
       "the opposite operation: SPLITTING Kilosort's mixed clusters.",
    11.5, False, INK)

out = r"D:\Gil\spike_sorting_agent\outputs\kilosort_variants_summary.pptx"
prs.save(out)
print("saved", out)
