---
layout: leetcode-entry
title: "920. Number of Music Playlists"
permalink: "/leetcode/problem/2023-08-06-920-number-of-music-playlists/"
leetcode_ui: true
entry_slug: "2023-08-06-920-number-of-music-playlists"
---

[920. Number of Music Playlists](https://leetcode.com/problems/number-of-music-playlists/description/) hard
[blog post](https://leetcode.com/problems/number-of-music-playlists/solutions/3870246/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/06082023-920-number-of-music-playlists?sd=pf)
![image.png](/assets/leetcode_daily_images/15955245.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/300

#### Problem TLDR

Playlists number playing `n` songs `goal` times, repeating each once in a `k` times

#### Intuition

We can search through the problem space, taking each new song with the given rules: song can be repeated only after another `k` song got played. When we have the `goal` songs, check if all distinct songs are played.

We can cache the solution by `curr` and `used` map, but that will give TLE.

The hard trick here is that the result only depends on how many distinct songs are played.

#### Approach

Use DFS and memo.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun numMusicPlaylists(n: Int, goal: Int, k: Int): Int {
        val cache = mutableMapOf<Pair<Int, Int>, Long>()
        fun dfs(curr: Int, used: Map<Int, Int>): Long = cache.getOrPut(curr to used.size) {
          if (curr > goal) {
            if ((1..n).all { used.contains(it) }) 1L else 0L
          } else (1..n).asSequence().map { i ->
              if (curr <= used[i] ?: 0) 0L else
                dfs(curr + 1, used.toMutableMap().apply { this[i] = curr + k })
            }.sum()!! % 1_000_000_007L
        }
        return dfs(1, mapOf()).toInt()
    }

```

