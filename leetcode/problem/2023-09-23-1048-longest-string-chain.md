---
layout: leetcode-entry
title: "1048. Longest String Chain"
permalink: "/leetcode/problem/2023-09-23-1048-longest-string-chain/"
leetcode_ui: true
entry_slug: "2023-09-23-1048-longest-string-chain"
---

[1048. Longest String Chain](https://leetcode.com/problems/longest-string-chain/description/) medium
[blog post](https://leetcode.com/problems/longest-string-chain/solutions/4079003/kotlin-graph/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23092023-1048-longest-string-chain?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/c592e2cd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/348

#### Problem TLDR

Longest chain of words with single character added

#### Intuition

We can build a graph, then use DFS to find a maximum depth.
To detect predecessor, we can use two pointers.

#### Approach

Careful with two pointers: iterate over short string and adjust the second pointer for long, not vice versa.

#### Complexity
- Time complexity:
$$O(w*n^2)$$, to build a graph

- Space complexity:
$$O(n^2)$$, for graph

#### Code

```kotlin

    fun longestStrChain(words: Array<String>): Int {
      fun isPred(a: String, b: String): Boolean {
        if (a.length != b.length - 1) return false
        var i = -1
        return !a.any {
          i++
          while (i < b.length && it != b[i]) i++
          i == b.length
        }
      }
      val fromTo = mutableMapOf<String, MutableSet<String>>()
      for (a in words)
        for (b in words)
          if (isPred(a, b))
            fromTo.getOrPut(a) { mutableSetOf() } += b
      val cache = mutableMapOf<String, Int>()
      fun dfs(w: String): Int = cache.getOrPut(w) {
        1 + (fromTo[w]?.map { dfs(it) }?.max() ?: 0)
      }
      return words.map { dfs(it) }?.max() ?: 0
    }

```

