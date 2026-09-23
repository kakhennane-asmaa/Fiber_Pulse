// ═══════════════════════════════════════════════════════════════════
// FiberPulse Service Worker
// استراتيجية: Network-first للـ HTML + Cache-first للأصول
// ═══════════════════════════════════════════════════════════════════

const CACHE_VERSION = 'fiberpulse-v1.0.1';
const CACHE_STATIC = CACHE_VERSION + '-static';
const CACHE_RUNTIME = CACHE_VERSION + '-runtime';

const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  'https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
  'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js'
];

self.addEventListener('install', (event) => {
  console.log('[FiberPulse SW] Installing', CACHE_VERSION);
  event.waitUntil(
    caches.open(CACHE_STATIC).then((cache) => {
      return Promise.all(
        STATIC_ASSETS.map((url) =>
          cache.add(url).catch((err) => console.warn('[SW] Failed:', url, err))
        )
      );
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  console.log('[FiberPulse SW] Activating', CACHE_VERSION);
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => !key.startsWith(CACHE_VERSION))
          .map((key) => {
            console.log('[FiberPulse SW] Deleting old cache:', key);
            return caches.delete(key);
          })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  if (request.mode === 'navigate' || url.pathname.endsWith('.html')) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_STATIC).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() =>
          caches.match(request).then((r) => r || caches.match('./index.html'))
        )
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          if (!response || response.status !== 200 || response.type === 'opaque') {
            return response;
          }
          const clone = response.clone();
          caches.open(CACHE_RUNTIME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => cached);
    })
  );
});

self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  if (event.data === 'GET_VERSION' && event.ports && event.ports[0]) {
    event.ports[0].postMessage({ version: CACHE_VERSION });
  }
});
