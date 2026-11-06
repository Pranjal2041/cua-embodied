// Readiness uses visible UI signals. The transport must outlast that wait.
const SETTLE_TIMEOUT_MS = 900000;
const PROTOCOL_TIMEOUT_MS = SETTLE_TIMEOUT_MS + 60000;

async function waitForReady(page, timeout = SETTLE_TIMEOUT_MS) {
  await page.waitForFunction(() => {
    const status = document.querySelector('#status');
    if (!status || ['Connecting', 'Processing'].includes(status.textContent)) return false;
    if (document.querySelector('#busy.visible')) return false;
    const images = [...document.querySelectorAll('img.visible')];
    if (images.some(im => !im.complete || !im.naturalWidth)) return false;
    const videos = [...document.querySelectorAll('video')].filter(v => v.getClientRects().length);
    return videos.every(v => !v.getAttribute('src') || v.error || (!v.seeking && v.readyState >= 2));
  }, {timeout, polling: 100});
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  return page.$eval('#status', el => el.textContent);
}

function describeError(error) {
  const messages = [];
  const seen = new Set();
  while (error && !seen.has(error)) {
    seen.add(error);
    messages.push(error.message || String(error));
    error = error.cause;
  }
  return messages.join(' | caused by: ');
}

module.exports = {SETTLE_TIMEOUT_MS, PROTOCOL_TIMEOUT_MS, waitForReady, describeError};
