---
layout: leetcode-entry
title: "2225. Find Players With Zero or One Losses"
permalink: "/leetcode/problem/2022-11-28-2225-find-players-with-zero-or-one-losses/"
leetcode_ui: true
entry_slug: "2022-11-28-2225-find-players-with-zero-or-one-losses"
---

[2225. Find Players With Zero or One Losses](https://leetcode.com/problems/find-players-with-zero-or-one-losses/) medium

[https://t.me/leetcode_daily_unstoppable/34](https://t.me/leetcode_daily_unstoppable/34)

```kotlin

    fun findWinners(matches: Array<IntArray>): List<List<Int>> {
        val winners = mutableMapOf<Int, Int>()
        val losers = mutableMapOf<Int, Int>()
        matches.forEach { (w, l) ->
            winners[w] = 1 + (winners[w]?:0)
            losers[l] = 1 + (losers[l]?:0)
        }
        return listOf(
            winners.keys
                .filter { !losers.contains(it) }
                .sorted(),
            losers
                .filter { (k, v) -> v == 1 }
                .map { (k, v) -> k}
                .sorted()
        )
    }

```

Just do what is asked.

O(NlogN) time, O(N) space

