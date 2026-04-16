const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

// ── Icon imports ──
const {
  FaQuestion, FaClock, FaExclamationTriangle, FaCheckCircle,
  FaBrain, FaStar, FaMicrophone, FaDatabase,
} = require("react-icons/fa");
const {
  MdTranslate, MdFingerprint, MdLabel, MdLanguage,
  MdBarChart, MdTrendingUp, MdVisibility, MdLightbulbOutline,
  MdRocketLaunch, MdAutorenew, MdWarning, MdStorage,
  MdTune, MdGTranslate, MdSpeed, MdChat,
} = require("react-icons/md");
const {
  HiOutlineSearchCircle, HiOutlineAdjustments,
} = require("react-icons/hi");

// ── Palette ──
const C = {
  navy:    "00355F",
  navyMid: "0A4A7A",
  teal:    "0891B2",
  amber:   "F59E0B",
  light:   "F0F9FF",
  white:   "FFFFFF",
  dark:    "1E293B",
  muted:   "64748B",
  green:   "16A34A",
  red:     "EF4444",
  indigo:  "6366F1",
  cardBg:  "FFFFFF",
  divider: "CBD5E1",
};

const FONT_H = "Trebuchet MS";
const FONT_B = "Calibri";

// ── Icon rendering ──
function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconPng(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// ── Shadow factory (fresh object each call) ──
const cardShadow = () => ({
  type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.10,
});

// ══════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════
async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.author = "Hackathon Team";
  pres.title = "Ask What Matters — Adaptive AI for Smarter Travel Reviews";

  // Pre-render icons
  const icons = {
    question:   await iconPng(FaQuestion,            "#" + C.amber, 256),
    clock:      await iconPng(FaClock,               "#" + C.teal,  256),
    warning:    await iconPng(FaExclamationTriangle,  "#" + C.red,   256),
    check:      await iconPng(FaCheckCircle,          "#" + C.green, 256),
    brain:      await iconPng(FaBrain,               "#" + C.teal,  256),
    star:       await iconPng(FaStar,                "#" + C.amber, 256),
    mic:        await iconPng(FaMicrophone,           "#" + C.teal,  256),
    db:         await iconPng(FaDatabase,             "#" + C.navy,  256),
    translate:  await iconPng(MdTranslate,            "#" + C.white, 256),
    fingerprint:await iconPng(MdFingerprint,          "#" + C.white, 256),
    label:      await iconPng(MdLabel,                "#" + C.white, 256),
    language:   await iconPng(MdLanguage,             "#" + C.white, 256),
    barChart:   await iconPng(MdBarChart,             "#" + C.teal,  256),
    trending:   await iconPng(MdTrendingUp,           "#" + C.amber, 256),
    visible:    await iconPng(MdVisibility,           "#" + C.teal,  256),
    bulb:       await iconPng(MdLightbulbOutline,     "#" + C.amber, 256),
    rocket:     await iconPng(MdRocketLaunch,         "#" + C.white, 256),
    refresh:    await iconPng(MdAutorenew,            "#" + C.white, 256),
    warnWhite:  await iconPng(MdWarning,              "#" + C.white, 256),
    storage:    await iconPng(MdStorage,              "#" + C.white, 256),
    tune:       await iconPng(MdTune,                 "#" + C.white, 256),
    gTranslate: await iconPng(MdGTranslate,           "#" + C.white, 256),
    speed:      await iconPng(MdSpeed,                "#" + C.teal,  256),
    chat:       await iconPng(MdChat,                 "#" + C.teal,  256),
    questionW:  await iconPng(FaQuestion,             "#" + C.white, 256),
    search:     await iconPng(HiOutlineSearchCircle,  "#" + C.teal,  256),
    adjustNav:  await iconPng(HiOutlineAdjustments,   "#" + C.white, 256),
  };

  // ── Slide number helper ──
  let slideNum = 0;
  function addSlideNumber(slide, dark = false) {
    slideNum++;
    slide.addText(String(slideNum), {
      x: 9.3, y: 5.25, w: 0.5, h: 0.3,
      fontSize: 10, fontFace: FONT_B,
      color: dark ? C.muted : "94A3B8",
      align: "right", margin: 0,
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 1 — Title
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };

    // Top accent bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal },
    });

    // Main title
    s.addText("Ask What Matters", {
      x: 0.8, y: 1.2, w: 8.4, h: 1.2,
      fontSize: 44, fontFace: FONT_H, color: C.white, bold: true,
      align: "left", margin: 0,
    });

    // Subtitle
    s.addText("Adaptive AI for Smarter Travel Reviews", {
      x: 0.8, y: 2.4, w: 8.4, h: 0.6,
      fontSize: 22, fontFace: FONT_B, color: C.teal,
      align: "left", margin: 0,
    });

    // Divider line
    s.addShape(pres.shapes.LINE, {
      x: 0.8, y: 3.2, w: 2.5, h: 0,
      line: { color: C.amber, width: 3 },
    });

    // Event info
    s.addText("2026 Wharton Hack-AI-thon  |  Expedia Group", {
      x: 0.8, y: 3.55, w: 8.4, h: 0.4,
      fontSize: 14, fontFace: FONT_B, color: C.muted,
      align: "left", margin: 0,
    });

    // Team
    s.addText("Sacha Bani Assad  |  Sam Bani Assad  |  Eli Trevino  |  Rachel Bao", {
      x: 0.8, y: 4.8, w: 8.4, h: 0.4,
      fontSize: 13, fontFace: FONT_B, color: C.muted,
      align: "left", margin: 0,
    });
    addSlideNumber(s, true);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 2 — The Problem
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("The Problem", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    s.addText("Reviews capture what travelers say \u2014 but miss what they don't.", {
      x: 0.7, y: 0.95, w: 8.6, h: 0.45,
      fontSize: 16, fontFace: FONT_B, color: C.muted, italic: true, margin: 0,
    });

    // Three stat cards
    const stats = [
      { num: "67%", label: "of sub-rating fields\nare blank across reviews", color: C.red },
      { num: "7,200", label: "reviews, most covering\nthe same 3\u20134 topics", color: C.amber },
      { num: "180+", label: "days since many fields\nwere last confirmed", color: C.teal },
    ];

    const cardW = 2.7, gap = 0.35, startX = 0.7;
    stats.forEach((st, i) => {
      const x = startX + i * (cardW + gap);
      // Card background
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 1.75, w: cardW, h: 2.2,
        fill: { color: C.white }, shadow: cardShadow(),
      });
      // Left accent bar
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 1.75, w: 0.07, h: 2.2,
        fill: { color: st.color },
      });
      // Big number
      s.addText(st.num, {
        x: x + 0.25, y: 1.95, w: cardW - 0.4, h: 0.9,
        fontSize: 48, fontFace: FONT_H, color: st.color, bold: true,
        align: "left", margin: 0,
      });
      // Label
      s.addText(st.label, {
        x: x + 0.25, y: 2.85, w: cardW - 0.4, h: 0.9,
        fontSize: 14, fontFace: FONT_B, color: C.dark,
        align: "left", margin: 0,
      });
    });

    // Bottom callout
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 4.25, w: 8.6, h: 0.7,
      fill: { color: C.navy },
    });
    s.addText("Standard review forms ask the same generic questions regardless of what\u2019s already known.", {
      x: 0.9, y: 4.25, w: 8.2, h: 0.7,
      fontSize: 14, fontFace: FONT_B, color: C.white, align: "center", valign: "middle",
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 3 — The Opportunity
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("The Opportunity", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    // Challenge question
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 1.1, w: 8.6, h: 0.7,
      fill: { color: C.navy },
    });
    s.addText("\u201CWhat information is unknown or outdated about a property, and what\u2019s the easiest way to learn it?\u201D", {
      x: 0.9, y: 1.1, w: 8.2, h: 0.7,
      fontSize: 15, fontFace: FONT_B, color: C.white, italic: true,
      align: "center", valign: "middle",
    });

    // Three gap types
    const gaps = [
      { icon: icons.question, accent: C.red,   title: "Missing",      desc: "No data exists \u2014 the field has never been\nrated or mentioned by any guest." },
      { icon: icons.clock,    accent: C.amber,  title: "Stale",        desc: "Data exists but may be outdated \u2014 recent\nreviews suggest conditions have changed." },
      { icon: icons.warning,  accent: C.indigo, title: "Contradicted", desc: "The listing claims one thing, but guest\nreviews consistently say otherwise." },
    ];

    gaps.forEach((g, i) => {
      const y = 2.15 + i * 1.05;
      // Card
      s.addShape(pres.shapes.RECTANGLE, {
        x: 0.7, y, w: 8.6, h: 0.9,
        fill: { color: C.white }, shadow: cardShadow(),
      });
      // Accent bar
      s.addShape(pres.shapes.RECTANGLE, {
        x: 0.7, y, w: 0.07, h: 0.9,
        fill: { color: g.accent },
      });
      // Icon
      s.addImage({ data: g.icon, x: 1.05, y: y + 0.2, w: 0.45, h: 0.45 });
      // Title
      s.addText(g.title, {
        x: 1.75, y: y + 0.05, w: 2.0, h: 0.4,
        fontSize: 18, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
      });
      // Desc
      s.addText(g.desc, {
        x: 1.75, y: y + 0.42, w: 7.3, h: 0.45,
        fontSize: 12.5, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 4 — Our Solution
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("Our Solution", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    s.addText("1\u20132 smart follow-up questions, tailored to each property, different every time.", {
      x: 0.7, y: 0.95, w: 8.6, h: 0.4,
      fontSize: 16, fontFace: FONT_B, color: C.muted, italic: true, margin: 0,
    });

    // Two-phase diagram
    const phaseY = 1.6;
    // Build phase box
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: phaseY, w: 4.0, h: 0.9,
      fill: { color: C.navy },
    });
    s.addText([
      { text: "BUILD", options: { bold: true, fontSize: 16, color: C.amber } },
      { text: "  Offline Enrichment", options: { fontSize: 14, color: C.white } },
    ], { x: 0.9, y: phaseY, w: 3.6, h: 0.45, fontFace: FONT_B, valign: "bottom", margin: 0 });
    s.addText("Vectorize, categorize, aggregate 7,200 reviews", {
      x: 0.9, y: phaseY + 0.45, w: 3.6, h: 0.4,
      fontSize: 11.5, fontFace: FONT_B, color: C.divider, margin: 0,
    });

    // Arrow (bold, larger, amber to stand out)
    s.addShape(pres.shapes.RECTANGLE, {
      x: 4.72, y: phaseY + 0.2, w: 0.56, h: 0.5,
      fill: { color: C.amber },
    });
    s.addText("\u2192", {
      x: 4.72, y: phaseY + 0.2, w: 0.56, h: 0.5,
      fontSize: 28, fontFace: FONT_B, color: C.white, bold: true, align: "center", valign: "middle",
    });

    // Ask phase box
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.3, y: phaseY, w: 4.0, h: 0.9,
      fill: { color: C.teal },
    });
    s.addText([
      { text: "ASK", options: { bold: true, fontSize: 16, color: C.white } },
      { text: "  Real-Time Follow-ups", options: { fontSize: 14, color: C.white } },
    ], { x: 5.5, y: phaseY, w: 3.6, h: 0.45, fontFace: FONT_B, valign: "bottom", margin: 0 });
    s.addText("Score gaps, select questions, parse answers", {
      x: 5.5, y: phaseY + 0.45, w: 3.6, h: 0.4,
      fontSize: 11.5, fontFace: FONT_B, color: C.white, margin: 0,
    });

    // Four differentiator cards (2x2 grid)
    const diffs = [
      { icon: icons.brain,   title: "Adaptive",    desc: "Questions change based on each property\u2019s current knowledge state" },
      { icon: icons.search,  title: "Diverse",     desc: "Cluster-diverse selection covers different property dimensions" },
      { icon: icons.visible, title: "Transparent",  desc: "Every question shows its full scoring rationale" },
      { icon: icons.chat,    title: "Low-Friction", desc: "Voice + text, max 20 words, no pleasantries" },
    ];

    diffs.forEach((d, i) => {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const x = 0.7 + col * 4.65;
      const y = 2.9 + row * 1.15;
      // Card
      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: 4.3, h: 1.0,
        fill: { color: C.white }, shadow: cardShadow(),
      });
      // Icon
      s.addImage({ data: d.icon, x: x + 0.2, y: y + 0.25, w: 0.45, h: 0.45 });
      // Title
      s.addText(d.title, {
        x: x + 0.8, y: y + 0.1, w: 3.2, h: 0.35,
        fontSize: 16, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
      });
      // Desc
      s.addText(d.desc, {
        x: x + 0.8, y: y + 0.48, w: 3.2, h: 0.45,
        fontSize: 12, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 5 — Data Foundation: Build Phase
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("Data Foundation: Build Phase", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    s.addText("Every review passes through a 4-step enrichment pipeline before it can inform question selection.", {
      x: 0.7, y: 0.95, w: 8.6, h: 0.4,
      fontSize: 14, fontFace: FONT_B, color: C.muted, margin: 0,
    });

    // 4 pipeline steps as horizontal cards
    const steps = [
      { icon: icons.language, num: "1", title: "Detect\nLanguage", detail: "langdetect\nDeterministic seed", bg: C.navy },
      { icon: icons.gTranslate, num: "2", title: "Translate\nto English", detail: "GPT-4.1-mini\ntemp = 0.0", bg: C.navyMid },
      { icon: icons.fingerprint, num: "3", title: "Compute\nEmbedding", detail: "text-embedding-\n3-small (384d)", bg: C.teal },
      { icon: icons.label, num: "4", title: "Tag Review\nTopics", detail: "28 topics\n11 clusters", bg: "0D9488" },
    ];

    const stepW = 2.05, stepGap = 0.2, stepStartX = 0.7;
    steps.forEach((st, i) => {
      const x = stepStartX + i * (stepW + stepGap);
      const y = 1.65;

      // Card bg
      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: stepW, h: 2.6,
        fill: { color: st.bg },
      });

      // Step number
      s.addText(st.num, {
        x: x + 0.15, y: y + 0.12, w: 0.4, h: 0.35,
        fontSize: 14, fontFace: FONT_B, color: st.bg, bold: true,
        fill: { color: C.white }, align: "center", valign: "middle",
      });

      // Icon
      s.addImage({ data: st.icon, x: x + (stepW - 0.55) / 2, y: y + 0.65, w: 0.55, h: 0.55 });

      // Title
      s.addText(st.title, {
        x: x + 0.1, y: y + 1.3, w: stepW - 0.2, h: 0.65,
        fontSize: 14, fontFace: FONT_H, color: C.white, bold: true, align: "center", margin: 0,
      });

      // Detail
      s.addText(st.detail, {
        x: x + 0.1, y: y + 1.95, w: stepW - 0.2, h: 0.55,
        fontSize: 11, fontFace: FONT_B, color: C.divider, align: "center", margin: 0,
      });

      // Arrow between steps
      if (i < 3) {
        s.addText("\u2192", {
          x: x + stepW - 0.05, y: y + 0.95, w: stepGap + 0.1, h: 0.5,
          fontSize: 22, fontFace: FONT_B, color: C.navy, align: "center", valign: "middle",
        });
      }
    });

    // Bottom callout
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 4.55, w: 8.6, h: 0.65,
      fill: { color: C.white }, shadow: cardShadow(),
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 4.55, w: 0.07, h: 0.65,
      fill: { color: C.amber },
    });
    s.addText("7,200 reviews enriched  \u00B7  12 properties  \u00B7  8 parallel workers  \u00B7  SHA-256 LLM cache for idempotent reruns", {
      x: 1.0, y: 4.55, w: 8.1, h: 0.65,
      fontSize: 12.5, fontFace: FONT_B, color: C.dark, valign: "middle", margin: 0,
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 6 — Knowledge Representation
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("Knowledge Representation", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    // LEFT COLUMN — EMA
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 1.15, w: 4.3, h: 3.5,
      fill: { color: C.white }, shadow: cardShadow(),
    });

    s.addText("Dual EMA Tracking", {
      x: 1.0, y: 1.25, w: 3.7, h: 0.4,
      fontSize: 18, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    s.addText([
      { text: "Short-term EMA", options: { bold: true, color: C.teal, breakLine: true } },
      { text: "Half-life = 5 reviews", options: { color: C.muted, breakLine: true } },
      { text: "Captures recent sentiment shifts", options: { color: C.dark, breakLine: true } },
      { text: "", options: { breakLine: true, fontSize: 8 } },
      { text: "Long-term EMA", options: { bold: true, color: C.amber, breakLine: true } },
      { text: "Half-life = 30 reviews", options: { color: C.muted, breakLine: true } },
      { text: "Historical baseline for comparison", options: { color: C.dark, breakLine: true } },
      { text: "", options: { breakLine: true, fontSize: 8 } },
      { text: "Drift Detection", options: { bold: true, color: C.red, breakLine: true } },
      { text: "When |short \u2212 long| > 0.5 with 5+ mentions,", options: { color: C.dark, breakLine: true } },
      { text: "the field is flagged as drifting \u2014", options: { color: C.dark, breakLine: true } },
      { text: "conditions have changed.", options: { color: C.dark } },
    ], {
      x: 1.0, y: 1.75, w: 3.7, h: 2.7,
      fontSize: 12.5, fontFace: FONT_B, margin: 0, paraSpaceAfter: 2,
    });

    // RIGHT COLUMN — Field breakdown
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.3, y: 1.15, w: 4.0, h: 3.5,
      fill: { color: C.white }, shadow: cardShadow(),
    });

    s.addText("55 Fields per Property", {
      x: 5.6, y: 1.25, w: 3.4, h: 0.4,
      fontSize: 18, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    // Field breakdown bars
    const fields = [
      { label: "Sub-ratings", count: 15, color: C.teal,   desc: "overall, cleanliness, service ..." },
      { label: "Schema fields", count: 12, color: C.amber, desc: "pet policy, amenities, check-in ..." },
      { label: "Guest topics", count: 28, color: C.indigo, desc: "WiFi, breakfast, pool, noise ..." },
    ];

    fields.forEach((f, i) => {
      const y = 1.85 + i * 0.85;
      const barW = (f.count / 55) * 3.0;
      // Bar
      s.addShape(pres.shapes.RECTANGLE, {
        x: 5.6, y, w: barW, h: 0.35,
        fill: { color: f.color },
      });
      // Count in bar
      s.addText(String(f.count), {
        x: 5.6, y, w: barW, h: 0.35,
        fontSize: 14, fontFace: FONT_H, color: C.white, bold: true,
        align: "center", valign: "middle",
      });
      // Label
      s.addText(f.label, {
        x: 5.6 + barW + 0.15, y, w: 3.0 - barW, h: 0.35,
        fontSize: 13, fontFace: FONT_H, color: C.dark, bold: true, valign: "middle", margin: 0,
      });
      // Desc
      s.addText(f.desc, {
        x: 5.6, y: y + 0.35, w: 3.4, h: 0.3,
        fontSize: 11, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });

    // Stored as
    s.addText("Each stored as a FieldState: value_known, EMA pair, mention count, last confirmed date", {
      x: 5.6, y: 4.15, w: 3.4, h: 0.4,
      fontSize: 10.5, fontFace: FONT_B, color: C.muted, margin: 0,
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 7 — Four-Factor Scoring
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("Four-Factor Scoring", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    s.addText("Every field scored by a weighted composite of four independent signals.", {
      x: 0.7, y: 0.95, w: 8.6, h: 0.35,
      fontSize: 14, fontFace: FONT_B, color: C.muted, margin: 0,
    });

    // Four scoring factors
    const factors = [
      { name: "Missing",        weight: "55%", barW: 5.5, color: C.red,    desc: "Is this field completely unknown? Highest priority." },
      { name: "Stale",          weight: "25%", barW: 2.5, color: C.amber,  desc: "Has information aged past 180 days, or has EMA drift been detected?" },
      { name: "Coverage Gap",   weight: "15%", barW: 1.5, color: C.teal,   desc: "Do few guests ever mention this? Inverse-frequency scoring." },
      { name: "Cross-Reference",weight: "20%", barW: 2.0, color: C.indigo, desc: "Do the property listing and guest reviews disagree?" },
    ];

    factors.forEach((f, i) => {
      const y = 1.45 + i * 0.7;

      // Weight label
      s.addText(f.weight, {
        x: 0.7, y, w: 0.7, h: 0.3,
        fontSize: 15, fontFace: FONT_H, color: f.color, bold: true, align: "right", valign: "middle", margin: 0,
      });

      // Factor name
      s.addText(f.name, {
        x: 1.55, y, w: 1.8, h: 0.3,
        fontSize: 13, fontFace: FONT_H, color: C.dark, bold: true, valign: "middle", margin: 0,
      });

      // Bar background
      s.addShape(pres.shapes.RECTANGLE, {
        x: 3.5, y: y + 0.03, w: 5.5, h: 0.24,
        fill: { color: "E2E8F0" },
      });

      // Bar fill
      s.addShape(pres.shapes.RECTANGLE, {
        x: 3.5, y: y + 0.03, w: f.barW, h: 0.24,
        fill: { color: f.color },
      });

      // Description
      s.addText(f.desc, {
        x: 1.55, y: y + 0.32, w: 7.5, h: 0.28,
        fontSize: 11, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });

    // Note about weights summing > 100%
    s.addText("Weights are intentionally > 100% \u2014 factors reinforce, not partition.", {
      x: 0.7, y: 4.3, w: 8.6, h: 0.25,
      fontSize: 10, fontFace: FONT_B, color: C.muted, italic: true, align: "center", margin: 0,
    });

    // Formula card
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 4.6, w: 8.6, h: 0.45,
      fill: { color: C.navy },
    });
    s.addText("score = 0.55 \u00D7 missing + 0.25 \u00D7 stale + 0.15 \u00D7 coverage + 0.20 \u00D7 cross_ref", {
      x: 0.9, y: 4.6, w: 8.2, h: 0.45,
      fontSize: 13, fontFace: "Consolas", color: C.white,
      align: "center", valign: "middle",
    });

    // Cluster diversity note
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 5.15, w: 8.6, h: 0.35,
      fill: { color: C.white }, shadow: cardShadow(),
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 5.15, w: 0.07, h: 0.35,
      fill: { color: C.teal },
    });
    s.addText("Cluster-diverse selection: Q1 and Q2 always come from different property dimensions.", {
      x: 1.0, y: 5.15, w: 8.1, h: 0.35,
      fontSize: 12, fontFace: FONT_B, color: C.dark, valign: "middle", margin: 0,
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 8 — The Live Experience
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("The Live Experience", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    // Left: Pipeline steps
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 1.15, w: 4.8, h: 4.0,
      fill: { color: C.white }, shadow: cardShadow(),
    });

    s.addText("11-Step Real-Time Pipeline", {
      x: 1.0, y: 1.25, w: 4.2, h: 0.4,
      fontSize: 16, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    const pipeSteps = [
      "Detect language",
      "Translate to English",
      "Compute 384d embedding",
      "Tag against 28-topic taxonomy",
      "Persist review + tags to SQLite",
      "Re-aggregate field states",
      "Rank all fields (4-factor scoring)",
      "Exclude already-covered topics",
      "Select 1\u20132 questions (cluster-diverse)",
      "Render natural-language questions",
      "Detect listing contradictions",
    ];

    pipeSteps.forEach((label, i) => {
      const y = 1.72 + i * 0.29;
      const numStr = String(i + 1);
      const circleW = numStr.length > 1 ? 0.3 : 0.22;  // wider for 2-digit
      // Step number circle
      s.addShape(pres.shapes.OVAL, {
        x: 1.02, y: y + 0.02, w: circleW, h: 0.22,
        fill: { color: i < 4 ? C.teal : (i < 7 ? C.amber : C.navy) },
      });
      s.addText(numStr, {
        x: 1.02, y: y + 0.02, w: circleW, h: 0.22,
        fontSize: numStr.length > 1 ? 8 : 9, fontFace: FONT_B, color: C.white, bold: true,
        align: "center", valign: "middle",
      });
      // Label
      s.addText(label, {
        x: 1.45, y, w: 3.75, h: 0.26,
        fontSize: 11.5, fontFace: FONT_B, color: C.dark, valign: "middle", margin: 0,
      });
    });

    // Right: Key features
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.8, y: 1.15, w: 3.5, h: 4.0,
      fill: { color: C.white }, shadow: cardShadow(),
    });

    s.addText("Key Features", {
      x: 6.1, y: 1.25, w: 2.9, h: 0.4,
      fontSize: 16, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    const features = [
      { icon: icons.speed, title: "< 3s latency", desc: "Full pipeline end-to-end" },
      { icon: icons.chat,  title: "Natural phrasing", desc: "LLM generates \u226420-word questions" },
      { icon: icons.mic,   title: "Voice + text input", desc: "Low friction for travelers" },
      { icon: icons.brain, title: "Smart parsing", desc: "Regex-first, LLM fallback" },
      { icon: icons.trending, title: "Feedback loop", desc: "Answers update states instantly" },
    ];

    features.forEach((f, i) => {
      const y = 1.75 + i * 0.65;
      s.addImage({ data: f.icon, x: 6.1, y: y + 0.03, w: 0.35, h: 0.35 });
      s.addText(f.title, {
        x: 6.6, y, w: 2.5, h: 0.25,
        fontSize: 13, fontFace: FONT_H, color: C.dark, bold: true, margin: 0,
      });
      s.addText(f.desc, {
        x: 6.6, y: y + 0.27, w: 2.5, h: 0.25,
        fontSize: 11, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 9 — Full Transparency
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.light };

    s.addText("Full Transparency", {
      x: 0.7, y: 0.35, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
    });

    s.addText("Every question comes with a complete explanation \u2014 no black box.", {
      x: 0.7, y: 0.95, w: 8.6, h: 0.35,
      fontSize: 14, fontFace: FONT_B, color: C.muted, italic: true, margin: 0,
    });

    // "Why This Question?" mock card
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 1.55, w: 8.6, h: 3.65,
      fill: { color: C.white }, shadow: cardShadow(),
    });

    // Header bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 1.55, w: 8.6, h: 0.55,
      fill: { color: C.navy },
    });
    s.addText("\u201CWhy This Question?\u201D \u2014 shown for every follow-up", {
      x: 0.9, y: 1.55, w: 8.2, h: 0.55,
      fontSize: 16, fontFace: FONT_H, color: C.white, bold: true, valign: "middle",
    });

    // Two columns of transparency features
    const leftFeatures = [
      { title: "Rank & Cluster", desc: "\"Rank #3 of 55 fields \u00B7 Room cluster\"" },
      { title: "Plain-English Reason", desc: "\"No reviews mention WiFi speed yet\"" },
      { title: "Score Breakdown Bars", desc: "Visual bars for each of the 4 scoring factors" },
      { title: "Raw Formula", desc: "0.55 \u00D7 1.00 + 0.25 \u00D7 0.00 + 0.15 \u00D7 0.92 + 0.20 \u00D7 1.00" },
    ];

    const rightFeatures = [
      { title: "Scoring Intermediates", desc: "Age: 42 days, drift detected, response rate: 3.5%" },
      { title: "Review Overlap", desc: "\"You mentioned: WiFi, Breakfast \u2014 excluded\"" },
      { title: "Runner-ups", desc: "#2 Bed Comfort (0.49) \u00B7 #3 Noise Level (0.42)" },
      { title: "Coverage Impact", desc: "\"Answering raises coverage from 45% \u2192 47%\"" },
    ];

    leftFeatures.forEach((f, i) => {
      const y = 2.3 + i * 0.7;
      s.addShape(pres.shapes.RECTANGLE, {
        x: 1.0, y, w: 0.06, h: 0.55,
        fill: { color: C.teal },
      });
      s.addText(f.title, {
        x: 1.25, y, w: 3.5, h: 0.25,
        fontSize: 12, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
      });
      s.addText(f.desc, {
        x: 1.25, y: y + 0.25, w: 3.5, h: 0.28,
        fontSize: 10.5, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });

    rightFeatures.forEach((f, i) => {
      const y = 2.3 + i * 0.7;
      s.addShape(pres.shapes.RECTANGLE, {
        x: 5.3, y, w: 0.06, h: 0.55,
        fill: { color: C.amber },
      });
      s.addText(f.title, {
        x: 5.55, y, w: 3.5, h: 0.25,
        fontSize: 12, fontFace: FONT_H, color: C.navy, bold: true, margin: 0,
      });
      s.addText(f.desc, {
        x: 5.55, y: y + 0.25, w: 3.5, h: 0.28,
        fontSize: 10.5, fontFace: FONT_B, color: C.muted, margin: 0,
      });
    });
    addSlideNumber(s);
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 10 — Impact & Next Steps (dark closing slide)
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };

    // Top accent bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal },
    });

    s.addText("Impact & Scalability", {
      x: 0.7, y: 0.3, w: 8.6, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true, margin: 0,
    });

    // LEFT column — Impact
    s.addText("Impact", {
      x: 0.7, y: 1.1, w: 4.0, h: 0.4,
      fontSize: 18, fontFace: FONT_H, color: C.amber, bold: true, margin: 0,
    });

    const impacts = [
      { icon: icons.refresh,  text: "Self-improving: each answer sharpens the next question" },
      { icon: icons.warnWhite, text: "Contradiction alerts surface listing inaccuracies" },
      { icon: icons.rocket,   text: "Richer property data with every single review" },
    ];

    impacts.forEach((item, i) => {
      const y = 1.55 + i * 0.6;
      s.addImage({ data: item.icon, x: 0.9, y: y + 0.05, w: 0.35, h: 0.35 });
      s.addText(item.text, {
        x: 1.45, y, w: 3.6, h: 0.5,
        fontSize: 13, fontFace: FONT_B, color: C.white, valign: "middle", margin: 0,
      });
    });

    // RIGHT column — Scalability
    s.addText("Scalability", {
      x: 5.5, y: 1.1, w: 4.0, h: 0.4,
      fontSize: 18, fontFace: FONT_H, color: C.teal, bold: true, margin: 0,
    });

    const scales = [
      { icon: icons.storage,  text: "SQLite \u2192 Postgres for production workloads" },
      { icon: icons.tune,     text: "YAML-configurable taxonomy and scoring weights" },
      { icon: icons.adjustNav, text: "SHA-256 keyed LLM cache \u2014 near-instant reruns" },
      { icon: icons.gTranslate, text: "Multilingual from day one (langdetect + translation)" },
    ];

    scales.forEach((item, i) => {
      const y = 1.55 + i * 0.6;
      s.addImage({ data: item.icon, x: 5.7, y: y + 0.05, w: 0.35, h: 0.35 });
      s.addText(item.text, {
        x: 6.25, y, w: 3.4, h: 0.5,
        fontSize: 13, fontFace: FONT_B, color: C.white, valign: "middle", margin: 0,
      });
    });

    // Divider — well below last scalability item
    s.addShape(pres.shapes.LINE, {
      x: 0.7, y: 4.0, w: 8.6, h: 0,
      line: { color: C.muted, width: 0.5 },
    });

    // Thank you
    s.addText("Thank You", {
      x: 0.7, y: 4.15, w: 8.6, h: 0.5,
      fontSize: 24, fontFace: FONT_H, color: C.white, bold: true, align: "center", margin: 0,
    });
    s.addText("Sacha Bani Assad  \u00B7  Sam Bani Assad  \u00B7  Eli Trevino  \u00B7  Rachel Bao", {
      x: 0.7, y: 4.65, w: 8.6, h: 0.35,
      fontSize: 13, fontFace: FONT_B, color: C.muted, align: "center", margin: 0,
    });
    s.addText("Questions?", {
      x: 0.7, y: 4.95, w: 8.6, h: 0.4,
      fontSize: 16, fontFace: FONT_B, color: C.teal, align: "center", margin: 0,
    });
    addSlideNumber(s, true);
  }

  // ── Write file ──
  const outPath = "Ask-What-Matters.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log(`\u2705 Generated ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
