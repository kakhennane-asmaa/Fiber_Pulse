// ═══════════════════════════════════════════════════════════════════
// FiberPulse Service Worker
// استراتيجية: Network-first للـ HTML + Cache-first للأصول
// ═══════════════════════════════════════════════════════════════════

const CACHE_VERSION = 'fiberpulse-v1.0.2';
const CACHE_STATIC = CACHE_VERSION + '-static';
const CACHE_RUNTIME = CACHE_VERSION + '-runtime';

const STATIC_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './sw.js'
];

self.addEventListener('install', (event) => {
  console.log('[FiberPulse SW] Installing', CACHE_VERSION);
  event.waitUntil(
    caches.open(CACHE_STATIC).then((cache) => {
      return Promise.all(STATIC_ASSETS.map(async (url) => {
        try {
          const request = new Request(url, { cache: 'reload' });
          await cache.add(request);
        } catch (err) {
          console.warn('[SW] Failed:', url, err);
        }
      }));
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
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;
  const isSameOrigin = url.origin === self.location.origin;

  if (request.mode === 'navigate' || (isSameOrigin && url.pathname.endsWith('.html'))) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response && response.ok && isSameOrigin) {
            const clone = response.clone();
            caches.open(CACHE_STATIC).then((cache) => cache.put(request, clone));
          }
          return response;
        })
        .catch(async () => (await caches.match(request)) || (await caches.match('./index.html')) || new Response('<h1>Offline</h1>', { status: 503, headers: { 'Content-Type': 'text/html; charset=utf-8' } }))
    );
    return;
  }

  event.respondWith(
    caches.match(request).then(async (cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          if (!response || response.status !== 200 || response.type === 'opaque' || !isSameOrigin) {
            return response;
          }
          const clone = response.clone();
          caches.open(CACHE_RUNTIME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => cached || new Response('', { status: 504, statusText: 'Offline' }));
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
