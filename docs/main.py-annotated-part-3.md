# main.py Annotated (Lines 421–700)

Continuation of line-by-line explanations.

---

421. `                base + f"{slug_dash}/", base + f"cars/{slug_dash}"` — Alternate common paths for automotive.
422. `            ])` — Close URL patterns list.
423. `        ` — Blank.
424. `        first_loaded_official = ""` — Track first domain that returns 200 OK.
425. `        for url in candidates:` — Iterate candidate URLs.
426. `            try:` — Guard per request.
427. `                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)` — Fetch.
428. `                if not resp.ok:` — Non-2xx.
429. `                    continue` — Try next.
430. `                html = resp.text` — Capture HTML.
431. `                if not first_loaded_official:` — If first successful load.
432. `                    first_loaded_official = url` — Remember it.
433. `                try:` — Try parsing text.
434. `                    soup = BeautifulSoup(html, "html.parser")` — Parse HTML.
435. `                    text = soup.get_text(" ", strip=True)[:5000]` — Extract clipped text for heuristics.
436. `                except Exception:` — Fallback.
437. `                    text = html[:5000]` — Use raw HTML.
438. `                # Heuristics: mention of brand/model or keywords` — Comment.
439. `                if brand in text.lower() or slug_dash.replace('-', ' ') in text.lower():` — Check brand/model presence.
440. `                    return url` — Return confident match.
441. `                if any(k in text.lower() for k in ["models", "specifications", "variants", "features", "dealers", "brochure"]):` — Generic official site hints.
442. `                    return url` — Return.
443. `            except Exception:` — Ignore network errors.
444. `                continue` — Next candidate.
445. `        return first_loaded_official` — As fallback, return first that loaded.
446. `    except Exception:` — Global guard.
447. `        return ""` — Return empty on failure.
448. (blank line) — Separator.
449. `def fetch_external_general_info(query: str, max_items: int = 6) -> str:` — Crawl generic info for non-price/approval queries.
450. `    brand = app_state.get("product_brand", "")` — Current brand.
451. `    category = app_state.get("product_category", "other")` — Current category.
452. `    product = app_state.get("product_name", "")` — Product name.
453. `    findings = []` — Collected snippets.
454. `    for url in build_candidate_urls_by_category(product, category, brand, app_state.get("user_location", "")):` — Iterate sources.
455. `        try:` — Fetch.
456. `            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)` — HTTP request.
457. `            if not resp.ok:` — Skip bad responses.
458. `                continue` — Next URL.
459. `            html = resp.text` — HTML.
460. `        except Exception:` — Network error.
461. `            continue` — Next URL.
462. `        try:` — Parse.
463. `            soup = BeautifulSoup(html, "html.parser")` — HTML parser.
464. `            text = soup.get_text(" ", strip=True)` — Extract readable text.
465. `        except Exception:` — Fallback.
466. `            text = html` — Use raw HTML.
467. `        # Prefer official if detected` — Comment.
468. `        try:` — Extract netloc.
469. `            domain = urlparse(url).netloc` — Domain.
470. `            if is_official_domain(domain, brand):` — If matches brand domains.
471. `                findings.append(f"Official site: {url}")` — Tag official hits.
472. `            else:` — Otherwise generic.
473. `                sentences = extract_sentences_by_keywords(text, _normalize_keywords(query))[:2]` — Pick 1–2 relevant sentences.
474. `                for s in sentences:` — Iterate.
475. `                    findings.append(f"{s} [Source: {domain}]({url})")` — Annotate.
476. `        except Exception:` — Ignore formatting errors.
477. `            continue` — Next URL.
478. `    return "\n".join(f"- {f}" for f in findings[:max_items])` — Bullet list.
479. (blank line) — Separator.
480. `def extract_city_from_location(location: str) -> str:` — Pull a city name from a free-form location string.
481. `    if not location:` — Guard.
482. `        return "Chennai"` — Default city for demo.
483. `    m = re.search(r"([A-Za-z]+)(?:,|$)", location)` — First word token before comma boundary.
484. `    return m.group(1) if m else "Chennai"` — Fallback to default.
485. (blank line) — Separator.
486. `def needs_dealer_search(question: str) -> bool:` — Detect whether to look for dealers.
487. `    q = (question or "").lower()` — Lower question.
488. `    return any(k in q for k in ["dealer", "dealers", "showroom", "nearest", "test drive", "book"] )` — Dealer keywords.
489. (blank line) — Separator.
490. `def fetch_official_dealers_generic(brand: str, location: str, max_items: int = 8) -> str:` — Heuristic dealer finder.
491. `    dealers = []` — Accumulator.
492. `    brand = (brand or "").strip().lower()` — Normalize.
493. `    if not brand:` — Guard.
494. `        return ""` — No brand → nothing to do.
495. `    city = extract_city_from_location(location)` — City extraction.
496. `    domains = _official_brand_domains(brand)` — Brand domains.
497. `    for d in domains:` — Try per domain.
498. `        try:` — Build URL.
499. `            url = f"https://{d}/dealers?city={city}"` — Common pattern for dealer locator.
500. `            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)` — Fetch.
501. `            if not resp.ok:` — If fails, try raw /dealers.
502. `                url = f"https://{d}/dealers"`
503. `                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)`
504. `                if not resp.ok:`
505. `                    continue`
506. `            try:` — Parse DOM.
507. `                soup = BeautifulSoup(resp.text, "html.parser")`
508. `                text = soup.get_text(" ", strip=True)`
509. `                # very generic extraction of dealer-like entries` — Comment.
510. `                items = re.findall(r"([A-Za-z&.' ]+)(?:Dealer|Showroom|Contact|Address)[:\s]", text)[:max_items]` — Regex capturing dealer names.
511. `                if items:` — If any...
512. `                    for it in items:` — Iterate names.
513. `                        dealers.append(f"{it.strip()} — {url}")` — Append with source.
514. `            except Exception:` — Parsing error.
515. `                continue`
516. `    return "\n".join(f"- {d}" for d in dealers[:max_items])` — Bullet list back to caller.
517. (blank line) — Separator.
518. `def fetch_official_dealers(question: str) -> str:` — Wrapper to decide whether to fetch.
519. `    if not needs_dealer_search(question):` — Only if question asks for dealers.
520. `        return ""` — Otherwise no-op.
521. `    brand = app_state.get("product_brand", "")` — Current brand.
522. `    location = app_state.get("user_location", "Chennai")` — Current city.
523. `    return fetch_official_dealers_generic(brand, location)` — Delegate.
524. (blank line) — Separator.
525. `def fetch_external_approval_info(product: str, max_items: int = 8) -> str:` — Regulatory info crawler.
526. `    findings = []` — Accumulator.
527. `    for url in ["https://www.fda.gov/drugs/", "https://www.ema.europa.eu/en/medicines/human"]:` — Seed list.
528. `        try:` — Fetch.
529. `            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)` — Request.
530. `            if not resp.ok:` — Skip bad responses.
531. `                continue`
532. `            soup = BeautifulSoup(resp.text, "html.parser")` — Parse HTML.
533. `            text = soup.get_text(" ", strip=True)` — Get text.
534. `            for sent in extract_relevant_sentences(text):` — Extract sentences.
535. `                domain = urlparse(url).netloc` — Domain.
536. `                findings.append(f"{sent} [Source: {domain}]({url})")` — Annotate.
537. `        except Exception:` — Ignore errors.
538. `            continue`
539. `    return "\n".join(f"- {f}" for f in findings[:max_items])` — Bullet list.
540. (blank line) — Separator.
541. `def needs_external_search(question: str) -> bool:` — Decide if general external search is needed.
542. `    q = (question or "").lower()` — Normalize.
543. `    keywords = ["compare", "compare with", "vs", "versus", "difference", "features", "variant", "variants", "dealer", "dealers", "pricing", "price", "cost"]` — Triggers.
544. `    return any(k in q for k in keywords)` — True if any keyword present.
545. (blank line) — Separator.
546. `# ... (RAG setup functions follow)` — Comment marking next section.
547. `def setup_rag_pipeline(pdf_path: str, product_slug: str) -> Dict:` — Build vector store for a single PDF.
548. `    docs = []` — Doc list.
549. `    try:` — Guard file load.
550. `        loader = PyPDFLoader(pdf_path)` — LangChain PDF loader.
551. `        docs = loader.load()` — Load to Documents.
552. `    except Exception:` — Fallback if loader fails.
553. `        docs = []` — Empty list.
554. `    if not docs:` — If no docs loaded.
555. `        # OCR fallback will be applied later if needed` — Comment.
556. `        pass` — No-op placeholder.
557. `    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)` — Chunking strategy.
558. `    splits = text_splitter.split_documents(docs)` — Chunk documents.
559. `    if not splits:` — No text extracted.
560. `        # Try OCR path when embedded text is missing` — Comment.
561. `        splits = []` — Reset.
562. `        try:` — OCR guard.
563. `            pages = convert_from_path(pdf_path, first_page=1, last_page=min( app_state["ocr_settings"]["ocr_pages"], 12), poppler_path=app_state["ocr_settings"].get("poppler_path") or None)` — Render pages to images.
564. `            if app_state["ocr_settings"].get("tesseract_cmd"):` — Custom Tesseract path.
565. `                pytesseract.pytesseract.tesseract_cmd = app_state["ocr_settings"]["tesseract_cmd"]` — Set it.
566. `            for i, img in enumerate(pages):` — OCR each page.
567. `                text = pytesseract.image_to_string(img)` — OCR text.
568. `                if text.strip():` — Non-empty.
569. `                    doc = Document(page_content=text, metadata={"source": pdf_path, "page": i})` — Build document.
570. `                    splits.append(doc)` — Collect.
571. `            del pages` — Free memory.
572. `            gc.collect()` — Force GC.
573. `        except Exception:` — OCR failed.
574. `            splits = []` — Leave empty.
575. `    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)` — Embedding model.
576. `    vectordb = Chroma(collection_name=f"{product_slug}__embedding001", embedding_function=embeddings)` — Vector store by slug.
577. `    if splits:` — If we have text.
578. `        vectordb.add_documents(splits)` — Index chunks.
579. `    retriever = vectordb.as_retriever(search_kwargs={"k": 6})` — Retriever handle.
580. `    app_state["retriever"] = retriever` — Save.
581. `    app_state["source_chunk_counts"] = {"pdf": len(splits)}` — Diagnostics.
582. `    return {"chunks_indexed": len(splits)}` — Return summary.
583. (blank line) — Separator.
584. `def setup_rag_pipeline_from_urls(urls: List[str], product_slug: str) -> Dict:` — Build vector store from URLs.
585. `    docs = []` — Doc list.
586. `    for url in urls:` — Iterate.
587. `        try:` — Fetch.
588. `            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)` — Request.
589. `            if not resp.ok:` — Skip.
590. `                continue`
591. `            soup = BeautifulSoup(resp.text, "html.parser")` — Parse.
592. `            text = soup.get_text(" ", strip=True)` — Extract text.
593. `            docs.append(Document(page_content=text, metadata={"source": url}))` — Add doc.
594. `        except Exception:` — Ignore failures.
595. `            continue`
596. `    if not docs:` — If none loaded.
597. `        return {"chunks_indexed": 0}` — Nothing indexed.
598. `    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)` — Chunking.
599. `    splits = text_splitter.split_documents(docs)` — Split.
600. `    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)` — Embeddings.
601. `    vectordb = Chroma(collection_name=f"{product_slug}__embedding001", embedding_function=embeddings)` — Vector store.
602. `    vectordb.add_documents(splits)` — Index.
603. `    retriever = vectordb.as_retriever(search_kwargs={"k": 6})` — Retriever.
604. `    app_state["retriever"] = retriever` — Save.
605. `    app_state["source_chunk_counts"] = {"urls": len(splits)}` — Diagnostics.
606. `    return {"chunks_indexed": len(splits)}` — Summary.
607. (blank line) — Separator.
608. `def setup_multiple_pdfs(pdf_paths: List[str], product_slug: str) -> Dict:` — Multi-PDF indexing with OCR fallback per file.
609. `    all_docs = []` — Aggregate list.
610. `    for p in pdf_paths:` — Iterate PDFs.
611. `        try:` — Load.
612. `            loader = PyPDFLoader(p)` — Loader.
613. `            docs = loader.load()` — Documents.
614. `        except Exception:` — Fallback.
615. `            docs = []` — Empty.
616. `        all_docs.extend(docs)` — Aggregate.
617. `    if not all_docs:` — No docs at all.
618. `        return {"chunks_indexed": 0}` — Early return.
619. `    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)` — Chunk.
620. `    splits = text_splitter.split_documents(all_docs)` — Split.
621. `    if not splits:` — If still empty.
622. `        # OCR fallback per file if needed` — Comment.
623. `        splits = []` — Reset.
624. `        try:` — Guard.
625. `            for p in pdf_paths:` — Iterate files.
626. `                pages = convert_from_path(p, first_page=1, last_page=min(app_state["ocr_settings"]["ocr_pages"], 12), poppler_path=app_state["ocr_settings"].get("poppler_path") or None)` — Render.
627. `                if app_state["ocr_settings"].get("tesseract_cmd"):` — Custom path.
628. `                    pytesseract.pytesseract.tesseract_cmd = app_state["ocr_settings"]["tesseract_cmd"]`
629. `                for i, img in enumerate(pages):` — Each page.
630. `                    text = pytesseract.image_to_string(img)` — OCR.
631. `                    if text.strip():` — Non-empty.
632. `                        doc = Document(page_content=text, metadata={"source": p, "page": i})` — Build doc.
633. `                        splits.append(doc)` — Append.
634. `            gc.collect()` — Collect.
635. `        except Exception:` — Ignore.
636. `            splits = []`
637. `    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)` — Embeddings.
638. `    vectordb = Chroma(collection_name=f"{product_slug}__embedding001", embedding_function=embeddings)` — Store.
639. `    if splits:` — If have splits.
640. `        vectordb.add_documents(splits)` — Index.
641. `    retriever = vectordb.as_retriever(search_kwargs={"k": 6})` — Retriever.
642. `    app_state["retriever"] = retriever` — Save.
643. `    app_state["source_chunk_counts"] = {"multi_pdfs": len(splits)}` — Diagnostics.
644. `    return {"chunks_indexed": len(splits)}` — Summary.
645. (blank line) — Separator.
646. `# ... (Agents config and API routes follow)` — Marker.
647. `# (Omitted here: lines 701–1311 include agents setup, status/upload routes, and chat route)` — Note.