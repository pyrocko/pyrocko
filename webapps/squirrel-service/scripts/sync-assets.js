
const fs = require('fs');
const path = require('path');

const targetBaseDir = path.join(__dirname, '..', 'page', 'vendor');

const rawAssets = [
  { from: 'node_modules/d3/dist/d3.min.js', to: 'd3.min.js' },
  { from: 'node_modules/vue/dist/vue.esm-browser.prod.js', to: 'vue.esm-browser.js' },
  { from: 'node_modules/vue-router/dist/vue-router.esm-browser.prod.js', to: 'vue-router.esm-browser.js' },
  { from: 'node_modules/quasar/dist/quasar.prod.css', to: 'quasar.css' },
  { from: 'node_modules/quasar/dist/quasar.umd.prod.js', to: 'quasar.umd.js' },
  { from: 'node_modules/leaflet/dist/leaflet.js', to: 'leaflet/leaflet.js' },
  { from: 'node_modules/leaflet/dist/leaflet.css', to: 'leaflet/leaflet.css' },
  { from: 'node_modules/leaflet/dist/leaflet.js.map', to: 'leaflet/leaflet.js.map' },
  { from: 'node_modules/leaflet/dist/images', to: 'leaflet/images' },
  { from: 'node_modules/@quasar/extras/exports/material-icons/material-icons.css', to: 'icons/material-icons.css' },
  { from: 'node_modules/@quasar/extras/exports/material-icons/web-font', to: 'icons/web-font' },
  { from: 'node_modules/@fontsource/roboto/index.css', to: 'fonts/roboto.css' },
  { from: 'node_modules/@fontsource/roboto/files', to: 'fonts/files', filter: (src, dest) => !(src.includes('vietnamese') || src.includes('cyrillic') || src.includes('greek')) },
  { from: 'node_modules/@vue-flow/core/dist/vue-flow-core.mjs', to: 'vue-flow/vue-flow-core.mjs'},
  { from: 'node_modules/@vue-flow/core/dist/style.css', to: 'vue-flow/style.css'},
  { from: 'node_modules/@vue-flow/core/dist/theme-default.css', to: 'vue-flow/theme-default.css'},
];

const assets = rawAssets.map(asset => ({
  src: path.join(__dirname, '..', asset.from),
  dest: path.join(targetBaseDir, asset.to),
  label: asset.to,
  filter: asset.filter
}));

// fs.mkdirSync(targetBaseDir, { recursive: true });

assets.forEach(asset => {
  fs.mkdirSync(path.dirname(asset.dest), { recursive: true });
  fs.cpSync(asset.src, asset.dest, { recursive: true, filter: asset.filter });
  console.log(`Copied: ${asset.label}`);
});

