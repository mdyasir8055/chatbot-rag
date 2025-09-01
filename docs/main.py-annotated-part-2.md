# main.py Annotated (Lines 201–420)

Literal, line-by-line explanation continued.

---

201. `extractor = ProductExtractor()` — Create a single extractor instance to reuse across requests.
202. (blank line) — Separator.
203. `# Pydantic models` — Comment for data models used in API payloads.
204. `class ChatMessage(BaseModel):` — Request schema for chat endpoint.
205. `    message: str` — The user’s chat text.
206. (blank line) — Separator.
207. `class SettingsUpdate(BaseModel):` — Request schema to update runtime settings.
208. `    user_location: Optional[str] = None` — Optional new location string.
209. `    disable_external: Optional[bool] = None` — Toggle external enrichment.
210. `    ocr_settings: Optional[Dict[str, Any]] = None` — Optional OCR config overrides.
211. (blank line) — Separator.
212. `class URLInput(BaseModel):` — Request schema for URL-based indexing.
213. `    urls: List[str]` — List of URLs to load.
214. (blank line) — Separator.
215. `class ChatResponse(BaseModel):` — Response schema for chat outputs.
216. `    response: str` — Final message back to the user.
217. `    used_external: bool = False` — Whether external enrichment was used.
218. (blank line) — Separator.
219. `# --- Helper functions (keeping all existing ones) ---` — Section header for helpers.
220. `def build_candidate_urls_by_category(product: str, category: str, brand: str = None, location: str = None) -> list:` — Builds seed URLs depending on category and params.
221. `    """Enhanced URL building based on category and parameters"""` — Docstring.
222. `    urls = []` — Start with empty list.
223. `    slug = product.lower().replace(' ', '-')` — Slugify product for URL path segments.
224. `    city = extract_city_from_location(location) if location else "Chennai"` — Extract city; default for demos.
225. (blank line) — Separator.
226. `    # Category-specific URL patterns` — Comment.
227. `    if category == 'automotive':` — Automotive branch.
228. (blank line) — Separator.
229. `        # Generic, brand-agnostic automotive sources` — Explain brand-agnostic design.
230. `        urls.extend([` — Add a batch of links.
231. `            f"https://en.wikipedia.org/wiki/{product}",` — Wikipedia page for product.
232. `            f"https://www.autotrader.com/cars-for-sale/{slug.split('-')[-1]}",` — Autotrader search on model token.
233. `            f"https://www.cars.com/shopping/results/?q={product.replace(' ', '+')}"` — Cars.com search with full query.
234. `        ])` — Close list.
235. `    elif category == 'pharmaceutical':` — Pharma sources.
236. `        urls.extend([` — Batch add.
237. `            f"https://www.drugs.com/{slug}.html",` — Drugs.com monograph guess.
238. `            f"https://www.fda.gov/drugs/",` — FDA drugs landing.
239. `            f"https://www.ema.europa.eu/en/medicines/human/EPAR/{slug}",` — EMA EPAR path guess.
240. `            f"https://en.wikipedia.org/wiki/{product}"` — Wikipedia article.
241. `        ])` — Close.
242. `    elif category == 'electronics':` — Electronics sources.
243. `        if brand:` — If brand known, try brand site path.
244. `            urls.extend([` — Add probable brand URLs and review sites.
245. `                f"https://www.{brand.lower()}.com/{slug}",` — Brand domain guess.
246. `                f"https://www.gsmarena.com/",` — Device database.
247. `                f"https://www.techradar.com/"` — Tech reviews.
248. `            ])` — Close.
249. `        urls.extend([` — Always add generic sources.
250. `            f"https://en.wikipedia.org/wiki/{product}",` — Wikipedia article.
251. `            f"https://www.amazon.com/s?k={product.replace(' ', '+')}"` — Amazon search.
252. `        ])` — Close.
253. `    else:` — Fallback for other categories.
254. `        # Fallback to general sources` — Comment.
255. `        urls.extend([` — Add generic links.
256. `            f"https://en.wikipedia.org/wiki/{product}",` — Wikipedia.
257. `            f"https://www.google.com/search?q={product.replace(' ', '+')}"` — Google search.
258. `        ])` — Close.
259. (blank line) — Separator.
260. `    random.shuffle(urls)` — Shuffle to distribute requests/avoid bias.
261. `    return urls` — Return list of candidates.
262. (blank line) — Separator.
263. `def infer_product_name_from_path(pdf_path: str) -> str:` — Heuristic from filename.
264. `    base = os.path.basename(pdf_path)` — Get basename (file name).
265. `    name = os.path.splitext(base)[0]` — Remove extension.
266. `    return name.replace('_', ' ').replace('-', ' ').strip()` — Normalize to human-readable.
267. (blank line) — Separator.
268. `def infer_product_name_from_content(documents: list) -> str:` — Guess product name from first couple pages.
269. `    try:` — Protect against malformed PDF content.
270. `        first_pages = []` — Collect first pages’ text.
271. `        for d in documents:` — Iterate documents.
272. `            p = d.metadata.get("page") if isinstance(d.metadata, dict) else None` — Read page index.
273. `            if p in (0, 1, None) and len(first_pages) < 2:` — Keep first two pages.
274. `                first_pages.append(d.page_content or "")` — Append page text.
275. `            if len(first_pages) >= 2:` — Stop early after two pages.
276. `                break` — Break loop.
277. `        blob = "\n".join(first_pages)[:8000]` — Merge and cap size.
278. `        if not blob:` — No text, abort.
279. `            return ""` — Return empty.
280. `        ` — Blank.
281. `        lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]` — Clean non-empty lines.
282. `        blacklist = {...}` — Words to exclude (e.g., brochure, guide).
283. `        candidates = []` — Candidate titles with scores.
284. `        for ln in lines:` — Iterate lines.
285. `            if 3 <= len(ln) <= 60 and any(c.isalpha() for c in ln):` — Keep plausible titles.
286. `                low = ln.lower()` — Lowercase cache.
287. `                if any(b in low for b in blacklist):` — Skip blacklisted.
288. `                    continue` — Continue.
289. `                caps = sum(1 for c in ln if c.isupper())` — Caps ratio heuristic.
290. `                score = caps / max(1, len(ln))` — Score by caps density.
291. `                if ln.istitle():` — Title-case bonus.
292. `                    score += 0.2` — Boost.
293. `                candidates.append((score, ln))` — Keep candidate.
294. `        ` — Blank.
295. `        seqs = re.findall(r"\\b(?:[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+){0,3})\\b", blob)` — Regex for capitalized sequences as fallback.
296. `        for s in seqs:` — Iterate sequences.
297. `            if 3 <= len(s) <= 40:` — Length guard.
298. `                candidates.append((0.15, s))` — Low score fallback candidates.
299. `        if not candidates:` — None found.
300. `            return ""` — Empty.
301. `        ` — Blank.
302. `        candidates.sort(key=lambda x: (abs(len(x[1].split())-2) <= 1, x[0]), reverse=True)` — Prefer 1–3 word titles, then score.
303. `        best = candidates[0][1]` — Pick top candidate text.
304. `        # Strip common separators and quotes from ends` — Comment.
305. `        best = best.strip('-â€":| "')` — Trim punctuation.
306. `        return best` — Return name.
307. `    except Exception:` — Robust fallback.
308. `        return ""` — Empty string on failure.
309. (blank line) — Separator.
310. `async def extract_pdf_text_fast(pdf_path: str) -> str:` — Quick text extraction from first 3 pages.
311. `    """Fast PDF text extraction - first 3 pages only"""` — Docstring.
312. `    try:` — Guard.
313. `        with open(pdf_path, 'rb') as file:` — Open file.
314. `            reader = PyPDF2.PdfReader(file)` — Create reader.
315. `            text = ""` — Accumulator.
316. `            # Only read first 3 pages for speed` — Comment.
317. `            for page_num in range(min(3, len(reader.pages))):` — Up to 3 pages.
318. `                text += reader.pages[page_num].extract_text()` — Extract text.
319. `                if len(text) > 2000:  # Stop if we have enough text` — Short-circuit.
320. `                    break` — Break loop.
321. `            return text` — Return.
322. `    except Exception:` — Any error.
323. `        return ""` — Empty string.
324. (blank line) — Separator.
325. `def extract_relevant_sentences(text: str) -> list:` — Pull regulation/approval sentences.
326. `    pattern = r"[^\.!?]{0,200}(?:FDA|Food and Drug Administration|EMA|approved|approval|indication)[^\.!?]{0,200}[\.!?]"` — Regex.
327. `    return list({s.strip() for s in re.findall(pattern, text, flags=re.I)})` — Unique trimmed matches.
328. (blank line) — Separator.
329. `def extract_price_sentences(text: str) -> list:` — Pull price-related sentences.
330. `    pattern = r"[^\.!?]{0,200}(?:price|pricing|msrp|mrp|cost|starts at|starting price|from|[$â‚¬Â£â‚¹]|INR|USD|EUR|GBP)[^\.!?]{0,200}[\.!?]"` — Regex.
331. `    return list({s.strip() for s in re.findall(pattern, text, flags=re.I)})` — Unique trimmed matches.
332. (blank line) — Separator.
333. `def fetch_external_pricing_info(product: str, max_items: int = 8) -> str:` — Crawl candidate sites for pricing.
334. `    findings = []` — Collected sentences.
335. `    category = app_state.get("product_category", "other")` — Use current category.
336. `    brand = app_state.get("product_brand", "")` — Current brand.
337. `    location = app_state.get("user_location", "")` — Current location.
338. (blank line) — Separator.
339. `    for url in build_candidate_urls_by_category(product, category, brand, location):` — Iterate candidate URLs.
340. `        try:` — Fetch guard.
341. `            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)` — Fetch with UA and timeout.
342. `            if not resp.ok:` — Skip bad responses.
343. `                continue` — Continue loop.
344. `            html = resp.text` — Page HTML.
345. `        except Exception:` — Network/parse error.
346. `            continue` — Skip URL.
347. `        try:` — Parse guard.
348. `            soup = BeautifulSoup(html, "html.parser")` — Parse HTML.
349. `            text = soup.get_text(separator=" ", strip=True)` — Extract visible text.
350. `        except Exception:` — Fallback if parsing fails.
351. `            text = html` — Use raw HTML.
352. `        for sent in extract_price_sentences(text):` — Iterate price sentences.
353. `            if len(findings) >= max_items:` — Stop when enough.
354. `                break` — Break inner loop.
355. `            try:` — Build annotated source label.
356. `                domain = urlparse(url).netloc` — Domain.
357. `                findings.append(f"{sent} [Source: {domain}]({url})")` — Append sentence with source.
358. `            except Exception:` — Ignore formatting error.
359. `                findings.append(sent)` — Append bare sentence.
360. `    return "\n".join(f"- {f}" for f in findings)` — Bullet list string.
361. (blank line) — Separator.
362. `def needs_external_pricing(question: str, context: str) -> bool:` — Decide if we should search the web for price.
363. `    q = (question or "").lower()` — Lowercased question.
364. `    ctx = (context or "")` — Context text.
365. `    keywords = [...]` — Price-related terms.
366. `    price_in_ctx = ("price" in ctx.lower()) or bool(re.search(r"[$â‚¬Â£â‚¹]\s?\d", ctx))` — If price already present.
367. `    return any(k in q for k in keywords) and not price_in_ctx` — Search only if asked and missing.
368. (blank line) — Separator.
369. `def _normalize_keywords(text: str) -> list:` — Token normalizer for keyword extraction.
370. `    text = (text or "").lower()` — Lowercase.
371. `    words = re.findall(r"[a-z0-9]+", text)` — Alnum tokens.
372. `    stop = {...}` — Stopword list.
373. `    return [w for w in words if w not in stop and len(w) > 2]` — Filter tokens.
374. (blank line) — Separator.
375. `def extract_sentences_by_keywords(text: str, keywords: list, max_chars: int = 400) -> list:` — Keyword-based sentence picker.
376. `    sentences = re.split(r"(?<=[.!?])\s+", text)` — Split on sentence boundaries.
377. `    keyset = set(keywords)` — For fast intersection.
378. `    picked = []` — Collector.
379. `    for s in sentences:` — Iterate sentences.
380. `        s_clean = s.strip()` — Trim whitespace.
381. `        if not s_clean or len(s_clean) > max_chars:` — Skip empty/long.
382. `            continue` — Next.
383. `        tokens = set(re.findall(r"[a-z0-9]+", s_clean.lower()))` — Tokenize.
384. `        if tokens & keyset:` — If any keyword present.
385. `            picked.append(s_clean)` — Keep.
386. `    return list(dict.fromkeys(picked))` — De-duplicate, preserve order.
387. (blank line) — Separator.
388. `def _official_brand_domains(brand: str) -> list:` — Generic official domain guesser.
389. `    """Guess official domains generically without brand-specific overrides."""` — Docstring.
390. `    b = (brand or "").strip().lower()` — Normalize brand text.
391. `    if not b:` — Guard empty.
392. `        return []` — No brand → no domains.
393. `    domains = [` — Candidates list.
394. `        f"www.{b}.com", f"{b}.com",` — .com variants.
395. `        f"www.{b}.co.in", f"{b}.co.in",` — .co.in variants (India).
396. `        f"www.{b}.in", f"{b}.in",` — .in variants.
397. `        f"www.{b}.co.uk", f"{b}.co.uk",` — .co.uk variants.
398. `    ]` — End list.
399. `    return list(dict.fromkeys(domains))` — De-duplicate preserving order.
400. (blank line) — Separator.
401. `def is_official_domain(domain: str, brand: str) -> bool:` — Determine if a domain belongs to brand.
402. `    host = (domain or "").lower()` — Normalize host.
403. `    allowed = _official_brand_domains(brand)` — Get candidates.
404. `    return any(host.endswith(d.replace("www.", "")) or host == d for d in allowed)` — Match against endings.
405. (blank line) — Separator.
406. `def guess_official_url(product: str) -> str:` — Try to find a live official URL for the product.
407. `    try:` — Guard network failures.
408. `        tokens = re.findall(r"[A-Za-z0-9]+", product or "")` — Tokenize product.
409. `        if not tokens:` — No tokens.
410. `            return ""` — Give up.
411. `        brand = tokens[0].lower()` — First token as brand guess.
412. `        slug_dash = "-".join(t.lower() for t in tokens)` — Hyphenated slug.
413. `        prod_low = (product or "").lower()` — Lowercased name.
414. (blank line) — Separator.
415. `        domains = _official_brand_domains(brand)` — Domain candidates.
416. `        candidates = []` — URL candidates.
417. `        for dom in domains:` — Build URLs for each domain.
418. `            base = f"https://{dom}/"` — Base scheme.
419. `            candidates.extend([` — Typical product URL patterns.
420. `                base, base + slug_dash, base + f"models/{slug_dash}",` — Home and model paths.