---
layout: leetcode-entry
title: "Determine If Two Strings Are Close"
permalink: "/leetcode/problem/2022-12-02-determine-if-two-strings-are-close/"
leetcode_ui: true
entry_slug: "2022-12-02-determine-if-two-strings-are-close"
---

[https://leetcode.com/problems/determine-if-two-strings-are-close/](https://leetcode.com/problems/determine-if-two-strings-are-close/) medium

[https://t.me/leetcode_daily_unstoppable/39](https://t.me/leetcode_daily_unstoppable/39)

```kotlin

    // cabbba -> c aa bbb -> 1 2 3
    // a bb ccc -> 1 2 3
    // uau
    // ssx
    fun closeStrings(word1: String, word2: String,
         f: (String) -> List<Int> = { it.groupBy { it }.values.map { it.size }.sorted() }
    ): Boolean = f(word1) == f(word2) && word1.toSet() == word2.toSet()

```

That is a simple task, you just need to know what exactly you asked for.
Space: O(n), Time: O(n)

