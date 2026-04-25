(function () {
  const dataNode = document.getElementById("leetcode-library-data");
  const resultsNode = document.getElementById("leetcode-results");
  if (!dataNode || !resultsNode) return;

  const DEFAULT_VISIBLE = 120;
  const VISIBLE_INCREMENT = 120;
  const filterKeys = ["query", "difficulty", "year", "language", "pattern"];

  const entries = JSON.parse(dataNode.textContent);
  const baseurl = resultsNode.dataset.baseurl || "";
  const searchInput = document.getElementById("leetcode-search");
  const difficultySelect = document.getElementById("leetcode-difficulty");
  const yearSelect = document.getElementById("leetcode-year");
  const languageSelect = document.getElementById("leetcode-language");
  const patternSelect = document.getElementById("leetcode-pattern");
  const metaNode = document.getElementById("leetcode-results-meta");
  const activeFiltersNode = document.getElementById("leetcode-active-filters");
  const emptyStateNode = document.getElementById("leetcode-empty-state");
  const resetButton = document.getElementById("leetcode-reset");
  const emptyResetButton = document.getElementById("leetcode-empty-reset");
  const loadMoreButton = document.getElementById("leetcode-load-more");
  const showAllButton = document.getElementById("leetcode-show-all");
  const randomLink = document.getElementById("leetcode-random-link");
  const cardViewButton = document.getElementById("leetcode-card-view");
  const listViewButton = document.getElementById("leetcode-list-view");
  const quickFilterButtons = Array.from(document.querySelectorAll("[data-leetcode-filter]"));
  const viewStorageKey = "leetcode-results-view";
  let visibleCount = DEFAULT_VISIBLE;
  let showAll = false;

  function withBase(path) {
    return `${baseurl}${path}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function titleCase(value) {
    return String(value).replace(/\b[a-z]/g, (letter) => letter.toUpperCase());
  }

  function chip(url, label) {
    return `<a href="${escapeHtml(url)}">${escapeHtml(label)}</a>`;
  }

  function renderEntry(entry) {
    const patternChips = (entry.pattern_slugs || [])
      .map((slug, index) => ({ slug, label: entry.pattern_labels[index] || slug }))
      .filter((pattern) => pattern.slug !== "implementation")
      .map((pattern) => {
        return `<a class="leetcode-pattern-chip" href="${escapeHtml(withBase(`/leetcode/pattern/${pattern.slug}/`))}">${escapeHtml(pattern.label)}</a>`;
      })
      .join("");

    const languageChips = (entry.languages || [])
      .map((language) => {
        return `<a class="leetcode-language-chip" href="${escapeHtml(withBase(`/leetcode/language/${language}/`))}">${escapeHtml(titleCase(language))}</a>`;
      })
      .join("");

    const aiTagChips = (entry.ai_tag_labels || [])
      .slice(0, 6)
      .map((label) => `<span class="leetcode-ai-chip">${escapeHtml(label)}</span>`)
      .join("");

    const links = [
      chip(withBase(entry.page_url), "Full post"),
      entry.problem_url ? chip(entry.problem_url, "LeetCode") : "",
      entry.kotlin_url ? chip(entry.kotlin_url, "Kotlin") : "",
      entry.rust_url ? chip(entry.rust_url, "Rust") : "",
      entry.youtube_url ? chip(entry.youtube_url, "YouTube") : "",
      entry.telegram_url ? chip(entry.telegram_url, "Telegram") : "",
    ].join("");

    return `
      <article class="leetcode-card leetcode-card--${escapeHtml(entry.difficulty)}">
        <div class="leetcode-card__top">
          <p class="leetcode-card__date">${escapeHtml(entry.display_date)}</p>
          <span class="leetcode-pill leetcode-pill--${escapeHtml(entry.difficulty)}">${escapeHtml(entry.difficulty)}</span>
        </div>
        <h3 class="leetcode-card__title">
          <a href="${escapeHtml(withBase(entry.page_url))}">${escapeHtml(entry.display_title)}</a>
        </h3>
        ${languageChips ? `<div class="leetcode-card__languages" aria-label="Languages">${languageChips}</div>` : ""}
        ${entry.takeaway ? `<p class="leetcode-card__takeaway">${escapeHtml(entry.takeaway)}</p>` : ""}
        ${patternChips ? `<div class="leetcode-card__patterns">${patternChips}</div>` : ""}
        ${aiTagChips ? `<div class="leetcode-card__ai-tags" aria-label="AI technique tags">${aiTagChips}</div>` : ""}
        <div class="leetcode-card__links">${links}</div>
      </article>
    `;
  }

  function matches(entry, filters) {
    const terms = filters.query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const haystack = [
      entry.problem_id,
      entry.slug,
      entry.display_title,
      entry.title,
      entry.takeaway,
      entry.difficulty,
      entry.year,
      (entry.pattern_labels || []).join(" "),
      (entry.pattern_slugs || []).join(" "),
      (entry.languages || []).join(" "),
      (entry.catalog_tags || []).join(" "),
      (entry.ai_tag_labels || []).join(" "),
      (entry.ai_tag_slugs || []).join(" "),
      entry.ai ? entry.ai.summary : "",
      entry.metrics ? entry.metrics.solution_line_bucket : "",
    ]
      .join(" ")
      .toLowerCase();

    if (terms.length > 0 && !terms.every((term) => haystack.includes(term))) return false;
    if (filters.difficulty && entry.difficulty !== filters.difficulty) return false;
    if (filters.year && entry.year !== filters.year) return false;
    if (filters.language && !(entry.languages || []).includes(filters.language)) return false;
    if (filters.pattern && !(entry.pattern_slugs || []).includes(filters.pattern)) return false;
    return true;
  }

  function emptyFilters() {
    return {
      query: "",
      difficulty: "",
      year: "",
      language: "",
      pattern: "",
    };
  }

  function currentFilters() {
    return {
      query: searchInput ? searchInput.value : "",
      difficulty: difficultySelect ? difficultySelect.value : "",
      year: yearSelect ? yearSelect.value : "",
      language: languageSelect ? languageSelect.value : "",
      pattern: patternSelect ? patternSelect.value : "",
    };
  }

  function setSelectValue(select, value) {
    if (!select) return;

    const requestedValue = String(value || "");
    const optionExists = !requestedValue || Array.from(select.options).some((option) => option.value === requestedValue);
    select.value = optionExists ? requestedValue : "";
  }

  function applyFilters(filters) {
    if (searchInput) searchInput.value = filters.query || "";
    setSelectValue(difficultySelect, filters.difficulty);
    setSelectValue(yearSelect, filters.year);
    setSelectValue(languageSelect, filters.language);
    setSelectValue(patternSelect, filters.pattern);
  }

  function hasActiveFilters(filters) {
    return filterKeys.some((key) => Boolean(filters[key]));
  }

  function selectedLabel(select, value) {
    if (!select || !value) return value;

    const option = Array.from(select.options).find((item) => item.value === value);
    return option ? option.textContent : value;
  }

  function activeFilterItems(filters) {
    const items = [];
    if (filters.query) items.push({ key: "query", label: `Search: ${filters.query}` });
    if (filters.difficulty) items.push({ key: "difficulty", label: `Difficulty: ${selectedLabel(difficultySelect, filters.difficulty)}` });
    if (filters.year) items.push({ key: "year", label: `Year: ${selectedLabel(yearSelect, filters.year)}` });
    if (filters.language) items.push({ key: "language", label: `Language: ${selectedLabel(languageSelect, filters.language)}` });
    if (filters.pattern) items.push({ key: "pattern", label: `Pattern: ${selectedLabel(patternSelect, filters.pattern)}` });
    return items;
  }

  function updateActiveFilters(filters) {
    if (!activeFiltersNode) return;

    const items = activeFilterItems(filters);
    activeFiltersNode.hidden = items.length === 0;
    activeFiltersNode.innerHTML = items
      .map((item) => {
        return `<button class="leetcode-active-filter" type="button" data-remove-filter="${escapeHtml(item.key)}">${escapeHtml(item.label)} <span aria-hidden="true">x</span></button>`;
      })
      .join("");
  }

  function readUrlFilters() {
    const filters = emptyFilters();
    if (!window.URLSearchParams) return filters;

    const params = new URLSearchParams(window.location.search);
    filters.query = params.get("q") || params.get("search") || "";
    filters.difficulty = params.get("difficulty") || "";
    filters.year = params.get("year") || "";
    filters.language = params.get("language") || "";
    filters.pattern = params.get("pattern") || "";
    return filters;
  }

  function writeUrlFilters(filters) {
    if (!window.history || !window.URL) return;

    const url = new URL(window.location.href);
    ["q", "search", "difficulty", "year", "language", "pattern"].forEach((key) => url.searchParams.delete(key));
    if (filters.query.trim()) url.searchParams.set("q", filters.query.trim());
    if (filters.difficulty) url.searchParams.set("difficulty", filters.difficulty);
    if (filters.year) url.searchParams.set("year", filters.year);
    if (filters.language) url.searchParams.set("language", filters.language);
    if (filters.pattern) url.searchParams.set("pattern", filters.pattern);
    window.history.replaceState(null, "", url);
  }

  function updateButtons(filteredCount, shownCount, activeFilters) {
    if (!loadMoreButton || !showAllButton) return;

    if (activeFilters || showAll || filteredCount <= shownCount) {
      loadMoreButton.hidden = true;
      showAllButton.hidden = true;
      return;
    }

    loadMoreButton.hidden = false;
    showAllButton.hidden = false;
    loadMoreButton.textContent = `Load ${Math.min(VISIBLE_INCREMENT, filteredCount - shownCount)} older entries`;
    showAllButton.textContent = `Show all ${filteredCount}`;
  }

  function savedView() {
    try {
      return window.localStorage ? window.localStorage.getItem(viewStorageKey) : "";
    } catch (error) {
      return "";
    }
  }

  function setSavedView(view) {
    try {
      if (window.localStorage) window.localStorage.setItem(viewStorageKey, view);
    } catch (error) {
      // Ignore storage failures; the visible state still updates.
    }
  }

  function setResultsView(view, options) {
    const normalizedView = view === "list" ? "list" : "cards";
    resultsNode.classList.toggle("leetcode-results--list", normalizedView === "list");

    if (cardViewButton) {
      cardViewButton.setAttribute("aria-pressed", String(normalizedView === "cards"));
      cardViewButton.classList.toggle("leetcode-button--ghost", normalizedView !== "cards");
    }

    if (listViewButton) {
      listViewButton.setAttribute("aria-pressed", String(normalizedView === "list"));
      listViewButton.classList.toggle("leetcode-button--ghost", normalizedView !== "list");
    }

    if (!options || options.persist !== false) setSavedView(normalizedView);
  }

  function render(options) {
    const shouldSyncUrl = options && options.syncUrl;
    const filters = currentFilters();
    const filtered = entries.filter((entry) => matches(entry, filters));
    const activeFilters = hasActiveFilters(filters);
    const visibleEntries = !activeFilters && !showAll ? filtered.slice(0, visibleCount) : filtered;

    resultsNode.innerHTML = visibleEntries.map(renderEntry).join("");
    resultsNode.hidden = visibleEntries.length === 0;
    if (emptyStateNode) emptyStateNode.hidden = visibleEntries.length > 0;

    if (activeFilters) {
      metaNode.textContent = `${filtered.length} matching entries.`;
    } else if (visibleEntries.length < filtered.length) {
      metaNode.textContent = `Showing ${visibleEntries.length} of ${filtered.length} entries.`;
    } else {
      metaNode.textContent = `Showing the full archive: ${filtered.length} entries.`;
    }

    if (resetButton) resetButton.disabled = !activeFilters;
    updateActiveFilters(filters);
    updateButtons(filtered.length, visibleEntries.length, activeFilters);
    if (shouldSyncUrl) writeUrlFilters(filters);
  }

  function resetFilters() {
    applyFilters(emptyFilters());
    visibleCount = DEFAULT_VISIBLE;
    showAll = false;
    render({ syncUrl: true });
  }

  [searchInput, difficultySelect, yearSelect, languageSelect, patternSelect]
    .filter(Boolean)
    .forEach((node) => {
      const eventName = node === searchInput ? "input" : "change";
      node.addEventListener(eventName, () => {
        visibleCount = DEFAULT_VISIBLE;
        showAll = false;
        render({ syncUrl: true });
      });
    });

  if (activeFiltersNode) {
    activeFiltersNode.addEventListener("click", (event) => {
      const button = event.target.closest("[data-remove-filter]");
      if (!button) return;

      const filters = currentFilters();
      filters[button.dataset.removeFilter] = "";
      applyFilters(filters);
      visibleCount = DEFAULT_VISIBLE;
      showAll = false;
      render({ syncUrl: true });
    });
  }

  quickFilterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const filters = emptyFilters();
      filterKeys.forEach((key) => {
        if (button.dataset[key] !== undefined) filters[key] = button.dataset[key];
      });
      applyFilters(filters);
      visibleCount = DEFAULT_VISIBLE;
      showAll = false;
      render({ syncUrl: true });
    });
  });

  if (resetButton) resetButton.addEventListener("click", resetFilters);
  if (emptyResetButton) emptyResetButton.addEventListener("click", resetFilters);

  if (loadMoreButton) {
    loadMoreButton.addEventListener("click", () => {
      visibleCount += VISIBLE_INCREMENT;
      render();
    });
  }

  if (showAllButton) {
    showAllButton.addEventListener("click", () => {
      showAll = true;
      render();
    });
  }

  if (cardViewButton) {
    cardViewButton.addEventListener("click", () => setResultsView("cards"));
  }

  if (listViewButton) {
    listViewButton.addEventListener("click", () => setResultsView("list"));
  }

  if (randomLink && entries.length > 0) {
    const randomEntry = entries[Math.floor(Math.random() * entries.length)];
    randomLink.href = withBase(randomEntry.page_url);
  }

  window.addEventListener("popstate", () => {
    applyFilters(readUrlFilters());
    visibleCount = DEFAULT_VISIBLE;
    showAll = false;
    render();
  });

  applyFilters(readUrlFilters());
  setResultsView(savedView(), { persist: false });
  render();
})();
