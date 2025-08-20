// server.js
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");
const sharp = require("sharp");
const ort = require("onnxruntime-node");

const app = express();
const upload = multer({ dest: "uploads/" });

const MODEL_PATH = path.join(__dirname, "models", "u2net.onnx");
const DATA_DIR = path.join(__dirname, "data");
const EXT_DB = path.join(DATA_DIR, "extensions.json");
const SIZE = 320; // MUST match model input

// Ensure folders exist
fs.mkdirSync("uploads", { recursive: true });
fs.mkdirSync("models", { recursive: true });
fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(EXT_DB)) fs.writeFileSync(EXT_DB, "[]", "utf8");

// ---------- Color conversions ----------
function rgb2xyz(r, g, b) {
  [r, g, b] = [r, g, b].map((v) => {
    v /= 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  });
  const x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  const y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
  const z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;
  return [x, y, z];
}

function xyz2lab(x, y, z) {
  const Xn = 0.95047,
    Yn = 1.0,
    Zn = 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x / Xn),
    fy = f(y / Yn),
    fz = f(z / Zn);
  const L = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const b = 200 * (fy - fz);
  return [L, a, b];
}

function rgb2lab(r, g, b) {
  return xyz2lab(...rgb2xyz(r, g, b));
}

// ---------- DeltaE for color matching ----------
function deltaE00(Lab1, Lab2) {
  const [L1, a1, b1] = Lab1;
  const [L2, a2, b2] = Lab2;
  const avgL = (L1 + L2) / 2;
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const avgC = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Math.pow(avgC, 7) / (Math.pow(avgC, 7) + Math.pow(25, 7))));
  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);
  let h1p = Math.atan2(b1, a1p) * (180 / Math.PI);
  if (h1p < 0) h1p += 360;
  let h2p = Math.atan2(b2, a2p) * (180 / Math.PI);
  if (h2p < 0) h2p += 360;
  const dLp = L2 - L1;
  const dCp = C2p - C1p;
  let dhp = h2p - h1p;
  if (dhp > 180) dhp -= 360;
  if (dhp < -180) dhp += 360;
  if (C1p * C2p === 0) dhp = 0;
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp * Math.PI) / 360);
  const avgHp = Math.abs(h1p - h2p) > 180 ? (h1p + h2p + 360) / 2 : (h1p + h2p) / 2;
  const T =
    1 -
    0.17 * Math.cos(((avgHp - 30) * Math.PI) / 180) +
    0.24 * Math.cos((2 * avgHp * Math.PI) / 180) +
    0.32 * Math.cos((3 * avgHp + 6) * Math.PI / 180) -
    0.2 * Math.cos(((4 * avgHp - 63) * Math.PI) / 180);
  const Sl = 1 + 0.015 * Math.pow(avgL - 50, 2) / Math.sqrt(20 + Math.pow(avgL - 50, 2));
  const Sc = 1 + 0.045 * avgC;
  const Sh = 1 + 0.015 * avgC * T;
  const Rt =
    -2 *
    Math.sqrt(Math.pow(avgC, 7) / (Math.pow(avgC, 7) + Math.pow(25, 7))) *
    Math.sin(((60 * Math.exp(-Math.pow((avgHp - 275) / 25, 2))) * Math.PI) / 180);
  return Math.sqrt(Math.pow(dLp / Sl, 2) + Math.pow(dCp / Sc, 2) + Math.pow(dHp / Sh, 2) + Rt * (dCp / Sc) * (dHp / Sh));
}

// ---------- ONNX Model ----------
let session = null;
async function loadModel() {
  if (!fs.existsSync(MODEL_PATH)) throw new Error(`ONNX model missing at ${MODEL_PATH}`);
  session = await ort.InferenceSession.create(MODEL_PATH, { graphOptimizationLevel: "all" });
  console.log("✅ ONNX model loaded");
}

// ---------- Preprocessing ----------
async function preprocessImageCHW(imagePath) {
  const { data, info } = await sharp(imagePath)
    .resize(SIZE, SIZE)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const chw = new Float32Array(3 * info.height * info.width);
  for (let i = 0; i < info.height * info.width; i++) {
    chw[i] = data[i * 3] / 255;
    chw[i + info.height * info.width] = data[i * 3 + 1] / 255;
    chw[i + 2 * info.height * info.width] = data[i * 3 + 2] / 255;
  }
  const rArr = [],
    gArr = [],
    bArr = [];
  for (let i = 0; i < data.length; i += 3) {
    rArr.push(data[i]);
    gArr.push(data[i + 1]);
    bArr.push(data[i + 2]);
  }
  return { tensor: new ort.Tensor("float32", chw, [1, 3, info.height, info.width]), H: info.height, W: info.width, rArr, gArr, bArr };
}

// ---------- Hair mask ----------
function getHairMask(outputTensor) {
  const dims = outputTensor.dims;
  const data = outputTensor.data;
  const [_, C, H, W] = dims;
  const mask = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) {
    if (C === 1) mask[i] = data[i] > 0.3 ? 1 : 0;
    else {
      let maxV = -Infinity;
      for (let c = 0; c < C; c++) if (data[c * H * W + i] > maxV) maxV = data[c * H * W + i];
      mask[i] = maxV > 0.3 ? 1 : 0;
    }
  }
  return { mask, H, W };
}

// ---------- Multi-shade detection ----------
function getDominantShades(rArr, gArr, bArr, mask, k = 12) {
  const pixels = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) pixels.push([rArr[i], gArr[i], bArr[i]]);
  if (pixels.length === 0) return [];

  // pick k random pixels as initial centroids
  let centroids = [];
  const used = new Set();
  while (centroids.length < Math.min(k, pixels.length)) {
    const idx = Math.floor(Math.random() * pixels.length);
    if (!used.has(idx)) {
      centroids.push([...pixels[idx]]);
      used.add(idx);
    }
  }

  let changed = true,
    iterations = 0;
  while (changed && iterations < 30) {
    const clusters = Array(centroids.length)
      .fill(0)
      .map(() => []);
    for (const p of pixels) {
      let minDist = Infinity,
        idx = 0;
      for (let c = 0; c < centroids.length; c++) {
        const [rC, gC, bC] = centroids[c];
        const dist = (p[0] - rC) ** 2 + (p[1] - gC) ** 2 + (p[2] - bC) ** 2;
        if (dist < minDist) {
          minDist = dist;
          idx = c;
        }
      }
      clusters[idx].push(p);
    }
    changed = false;
    for (let c = 0; c < centroids.length; c++) {
      if (clusters[c].length === 0) continue;
      const rAvg = clusters[c].reduce((s, p) => s + p[0], 0) / clusters[c].length;
      const gAvg = clusters[c].reduce((s, p) => s + p[1], 0) / clusters[c].length;
      const bAvg = clusters[c].reduce((s, p) => s + p[2], 0) / clusters[c].length;
      if (rAvg !== centroids[c][0] || gAvg !== centroids[c][1] || bAvg !== centroids[c][2]) {
        centroids[c] = [rAvg, gAvg, bAvg];
        changed = true;
      }
    }
    iterations++;
  }

  return centroids.map(([r, g, b]) => ({
    hex: `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`,
    lab: rgb2lab(r, g, b),
  }));
}

// ---------- Endpoints ----------
app.get("/health", (req, res) => res.json({ ok: true, modelLoaded: !!session }));

app.post("/analyze-user", upload.single("file"), async (req, res) => {
  try {
    if (!session) await loadModel();
    const filePath = req.file.path;
    const { tensor, H, W, rArr, gArr, bArr } = await preprocessImageCHW(filePath);
    const results = await session.run({ [session.inputNames[0]]: tensor });
    const output = results[session.outputNames[0]];
    const maskData = getHairMask(output);
    const hairShades = getDominantShades(rArr, gArr, bArr, maskData.mask, 3);
    fs.unlinkSync(filePath);
    res.json({ hair: hairShades });
  } catch (err) {
    console.error(err);
    if (req.file?.path && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post("/match", upload.single("file"), async (req, res) => {
  try {
    if (!session) await loadModel();
    const filePath = req.file.path;
    const { tensor, H, W, rArr, gArr, bArr } = await preprocessImageCHW(filePath);
    const results = await session.run({ [session.inputNames[0]]: tensor });
    const output = results[session.outputNames[0]];
    const maskData = getHairMask(output);
    const hairShades = getDominantShades(rArr, gArr, bArr, maskData.mask, 3);
    fs.unlinkSync(filePath);

    const db = JSON.parse(fs.readFileSync(EXT_DB, "utf8"));
    const topMatches = hairShades.map((shade) =>
      db
        .map((ext) => ({ ...ext, deltaE: deltaE00(shade.lab, ext.lab) }))
        .sort((a, b) => a.deltaE - b.deltaE)
        .slice(0, Number(req.query.k || 3))
    );

    res.json({ userHair: hairShades, topMatches });
  } catch (err) {
    console.error(err);
    if (req.file?.path && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post("/upload-extension", upload.single("file"), async (req, res) => {
  try {
    const id = (req.body.id || "").trim();
    const name = (req.body.name || "").trim();
    if (!id || !name) {
      if (req.file?.path) fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: "Provide id and name" });
    }
    const { data, info } = await sharp(req.file.path).resize(256, 256).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    let r = 0,
      g = 0,
      b = 0;
    for (let i = 0; i < data.length; i += 3) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
    }
    const n = data.length / 3;
    r /= n;
    g /= n;
    b /= n;
    const [L, a, bLab] = rgb2lab(r, g, b);
    const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
    const hex = `#${[r,g,b].map(clamp).map(v => v.toString(16).padStart(2,"0")).join("")}`;
    const dest = path.join(DATA_DIR, `${id}.png`);
    await fsp.rename(req.file.path, dest);
    const db = JSON.parse(fs.readFileSync(EXT_DB, "utf8"));
    const idx = db.findIndex((x) => x.id === id);
    const rec = { id, name, hex, lab: [L, a, bLab], image: `data/${id}.png` };
    if (idx >= 0) db[idx] = rec;
    else db.push(rec);
    fs.writeFileSync(EXT_DB, JSON.stringify(db, null, 2), "utf8");
    res.json({ saved: rec });
  } catch (err) {
    console.error(err);
    if (req.file?.path && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.get("/extensions", (req, res) => {
  const db = JSON.parse(fs.readFileSync(EXT_DB, "utf8"));
  res.json(db);
});

app.use("/data", express.static(DATA_DIR));

const PORT = Number(process.env.PORT || 3000);
app.listen(PORT, async () => {
  try {
    await loadModel();
  } catch (e) {
    console.error("Model not loaded yet:", e.message);
  }
  console.log(`Node hair matcher running on http://localhost:${PORT}`);
});
