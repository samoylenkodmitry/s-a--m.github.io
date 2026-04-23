---
layout: leetcode-entry
title: "1220. Count Vowels Permutation"
permalink: "/leetcode/problem/2023-10-28-1220-count-vowels-permutation/"
leetcode_ui: true
entry_slug: "2023-10-28-1220-count-vowels-permutation"
---

[1220. Count Vowels Permutation](https://leetcode.com/problems/count-vowels-permutation/description/) hard
[blog post](https://leetcode.com/problems/count-vowels-permutation/solutions/4216643/kotlin-dfs-memo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28102023-1220-count-vowels-permutation?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/414f430b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/384

#### Problem TLDR

Count of `n` lengths paths according to graph rules `a`->`e`, `e`->(`a`, `i`), etc

#### Intuition

This is a straghtforward DFS + memoization dynamic programming problem. Given the current position and the previous character, we know the suffix answer. It is independent of any other factors, so can be cached.

#### Approach

Let's write DFS + memo
* use Kotlin's `sumOf` API

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countVowelPermutation(n: Int): Int {
      val vs = mapOf('a' to arrayOf('e'),
                     'e' to arrayOf('a', 'i'),
                     'i' to arrayOf('a', 'e', 'o', 'u'),
                     'o' to arrayOf('i', 'u'),
                     'u' to arrayOf('a'),
                     '.' to arrayOf('a', 'e', 'i', 'o', 'u'))
      val dp = mutableMapOf<Pair<Int, Char>, Long>()
      fun dfs(i: Int, c: Char): Long = if (i == n) 1L else
        dp.getOrPut(i to c) { vs[c]!!.sumOf { dfs(i + 1, it) } } %
        1_000_000_007L
      return dfs(0, '.').toInt()
    }

```
Iterative version
![image.png](/assets/leetcode_daily_images/db81b9bb.webp)
Another one-liner
![image.png](/assets/leetcode_daily_images/85fca810.webp)

