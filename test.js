// hair-matcher.js (CommonJS)
const path = require("path");
const ort = require("onnxruntime-node");
const sharp = require("sharp");

const MODEL_PATH = path.resolve(__dirname, "models/u2net.onnx");
const HAIR_CLASS = 17; // hair class index
const SIZE = 320;      // U2Net input size

(async () => {
  const imagePath = process.argv[2] || path.resolve(__dirname, "test.jpg");
  console.log("Using image:", imagePath);

  // Load ONNX model
  const session = await ort.InferenceSession.create(MODEL_PATH);

  // Grab the actual input/output names from metadata
  const inputName = session.inputMetadata[0].name;  // 'input.1'
  const outputName = session.outputMetadata[0].name; // '1959'
  console.log("Using input:", inputName, "| output:", outputName);

  // Preprocess image
  const inputTensor = await preprocessImage(imagePath, SIZE, SIZE);

  // Feed model with correct input name
  const results = await session.run({ [inputName]: inputTensor });
  const output = results[outputName];

  console.log("Output dims:", output.dims);
  console.log("Sample values:", Array.from(output.data.slice(0, 20)));

  // Get hair mask and average color
  const seg = getHairMask(output.data, output.dims, HAIR_CLASS);
  const avgColor = await averageColor(imagePath, seg, HAIR_CLASS);

  console.log("Average hair color (RGB):", avgColor);

})().catch(err => {
  console.error(err);
  process.exit(1);
});

async function preprocessImage(imagePath, w, h) {
  const { data } = await sharp(imagePath)
    .resize(w, h)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const f32 = new Float32Array(3 * w * h);
  for (let i = 0; i < w * h; i++) {
    f32[i] = data[i * 3] / 255;
    f32[i + w * h] = data[i * 3 + 1] / 255;
    f32[i + 2 * w * h] = data[i * 3 + 2] / 255;
  }
  return new ort.Tensor("float32", f32, [1, 3, h, w]);
}

function getHairMask(data, dims, hairClass) {
  const [ , C, H, W ] = dims;

  if (C === 1) {
    const mask = new Int32Array(H * W);
    for (let i = 0; i < H * W; i++) mask[i] = data[i] > 0.3 ? hairClass : 0;
    return { mask, w: W, h: H };
  }

  const mask = new Int32Array(H * W);
  for (let i = 0; i < H * W; i++) {
    let maxVal = -Infinity, maxIdx = 0;
    for (let c = 0; c < C; c++) {
      const v = data[c * H * W + i];
      if (v > maxVal) { maxVal = v; maxIdx = c; }
    }
    mask[i] = maxIdx;
  }
  return { mask, w: W, h: H };
}

async function averageColor(imagePath, { mask, w, h }, classIndex) {
  const { data } = await sharp(imagePath).resize(w, h).raw().toBuffer({ resolveWithObject: true });

  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < w * h; i++) {
    if (mask[i] === classIndex) {
      r += data[i * 3];
      g += data[i * 3 + 1];
      b += data[i * 3 + 2];
      n++;
    }
  }
  if (!n) return { r: 0, g: 0, b: 0, count: 0 };
  return { r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n), count: n };
}
