---
layout: leetcode-entry
title: "1207. Unique Number of Occurrences"
permalink: "/leetcode/problem/2022-11-30-1207-unique-number-of-occurrences/"
leetcode_ui: true
entry_slug: "2022-11-30-1207-unique-number-of-occurrences"
---

[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/) easy

[https://t.me/leetcode_daily_unstoppable/36](https://t.me/leetcode_daily_unstoppable/36)

```kotlin

fun uniqueOccurrences(arr: IntArray): Boolean {
	val counter = mutableMapOf<Int, Int>()
	arr.forEach { n -> counter[n] = 1 + (counter[n] ?: 0) }
	val freq = mutableSetOf<Int>()
	return !counter.values.any { count -> !freq.add(count) }
}

```

Nothing interesting, just count and filter.

O(N) time, O(N) space

