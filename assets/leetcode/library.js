(function () {
  const dataNode = document.getElementById("leetcode-library-data");
  const resultsNode = document.getElementById("leetcode-results");
  if (!dataNode || !resultsNode) return;

  const entries = JSON.parse(dataNode.textContent);
  const baseurl = resultsNode.dataset.baseurl || "";
  const searchInput = document.getElementById("leetcode-search");
  const difficultySelect = document.getElementById("leetcode-difficulty");
  const yearSelect = document.getElementById("leetcode-year");
  const languageSelect = document.getElementById("leetcode-language");
  const patternSelect = document.getElementById("leetcode-pattern");
  const metaNode = document.getElementById("leetcode-results-meta");

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

    const links = [
      chip(withBase(entry.page_url), "Entry"),
      entry.problem_url ? chip(entry.problem_url, "LeetCode") : "",
      entry.kotlin_url ? chip(entry.kotlin_url, "Kotlin") : "",
      entry.rust_url ? chip(entry.rust_url, "Rust") : "",
      entry.youtube_url ? chip(entry.youtube_url, "YouTube") : "",
      entry.telegram_url ? chip(entry.telegram_url, "Telegram") : "",
    ].join("");

    return `
      <article class="leetcode-card">
        <div class="leetcode-card__top">
          <p class="leetcode-card__date">${escapeHtml(entry.display_date)}</p>
          <span class="leetcode-pill leetcode-pill--${escapeHtml(entry.difficulty)}">${escapeHtml(entry.difficulty)}</span>
        </div>
        <h3 class="leetcode-card__title">
          <a href="${escapeHtml(withBase(entry.page_url))}">${escapeHtml(entry.display_title)}</a>
        </h3>
        ${entry.takeaway ? `<p class="leetcode-card__takeaway">${escapeHtml(entry.takeaway)}</p>` : ""}
        ${patternChips ? `<div class="leetcode-card__patterns">${patternChips}</div>` : ""}
        <div class="leetcode-card__links">${links}</div>
      </article>
    `;
  }

  function matches(entry, filters) {
    const query = filters.query.trim().toLowerCase();
    const haystack = [
      entry.display_title,
      entry.takeaway,
      entry.difficulty,
      entry.year,
      (entry.pattern_labels || []).join(" "),
      (entry.pattern_slugs || []).join(" "),
      (entry.languages || []).join(" "),
    ]
      .join(" ")
      .toLowerCase();

    if (query && !haystack.includes(query)) return false;
    if (filters.difficulty && entry.difficulty !== filters.difficulty) return false;
    if (filters.year && entry.year !== filters.year) return false;
    if (filters.language && !(entry.languages || []).includes(filters.language)) return false;
    if (filters.pattern && !(entry.pattern_slugs || []).includes(filters.pattern)) return false;
    return true;
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

  function render() {
    const filters = currentFilters();
    const filtered = entries.filter((entry) => matches(entry, filters));
    resultsNode.innerHTML = filtered.map(renderEntry).join("");
    metaNode.textContent = `${filtered.length} matching entries.`;
  }

  [searchInput, difficultySelect, yearSelect, languageSelect, patternSelect]
    .filter(Boolean)
    .forEach((node) => {
      const eventName = node === searchInput ? "input" : "change";
      node.addEventListener(eventName, render);
    });

  render();
})();
