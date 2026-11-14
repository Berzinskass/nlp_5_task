#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (pandoc -> HTML -> Chromium print-to-PDF)
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md -s --embed-resources --resource-path=..:. --css style.css \
  --metadata title="Emotion Classification of Lithuanian Texts" -o report.html
node -e '
const { chromium } = require("playwright");
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto("file://" + process.cwd() + "/report.html", { waitUntil: "networkidle" });
  await p.pdf({ path: "report.pdf", format: "A4", printBackground: true,
    margin: { top: "22mm", bottom: "22mm", left: "25mm", right: "20mm" },
    displayHeaderFooter: true, headerTemplate: "<span></span>",
    footerTemplate: "<div style=\"font-size:8pt;width:100%;text-align:center;color:#666\"><span class=\"pageNumber\"></span></div>" });
  await b.close();
})();'
echo "built report/report.pdf"
