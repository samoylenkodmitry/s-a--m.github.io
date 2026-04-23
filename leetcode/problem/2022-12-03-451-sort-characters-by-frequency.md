---
layout: leetcode-entry
title: "451. Sort Characters By Frequency"
permalink: "/leetcode/problem/2022-12-03-451-sort-characters-by-frequency/"
leetcode_ui: true
entry_slug: "2022-12-03-451-sort-characters-by-frequency"
---

[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/) medium

[https://t.me/leetcode_daily_unstoppable/40](https://t.me/leetcode_daily_unstoppable/40)

```kotlin

    fun frequencySort(s: String): String =
        s.groupBy { it }
        .values
        .map { it to it.size }
        .sortedBy { -it.second }
        .map { it.first }
        .flatten()
        .joinToString("")

```

Very simple task, can be written in a functional style.
Space: O(n), Time: O(n)

