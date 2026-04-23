---
layout: leetcode-entry
title: "2707. Extra Characters in a String"
permalink: "/leetcode/problem/2023-09-02-2707-extra-characters-in-a-string/"
leetcode_ui: true
entry_slug: "2023-09-02-2707-extra-characters-in-a-string"
---

[2707. Extra Characters in a String](https://leetcode.com/problems/extra-characters-in-a-string/description/) medium
[blog post](https://leetcode.com/problems/extra-characters-in-a-string/solutions/3990697/kotlin-trie-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/2092023-2707-extra-characters-in?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/e2ce6773.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/327

#### Problem TLDR

Min count of leftovers after string split by the dictionary

#### Intuition

We can search all possible splits at every position when we find a word. To quickly find a word, let's use a `Trie`. The result will only depend on the suffix of the string, so can be cached.

#### Approach

Do DFS, each time compare a `skipped` result with any `take_word` result, if found a word. We must continue to search, because some words can be prefixes to others: `leet`, `leetcode` -> `leetcodes`, taking `leet` is not optimal.

#### Complexity

- Time complexity:
$$O(n^2)$$, DFS depth is `n` and another `n` for the inner iteration

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    class Trie(var w: Boolean = false) : HashMap<Char, Trie>()
    fun minExtraChar(s: String, dictionary: Array<String>): Int {
      val trie = Trie()
      for (w in dictionary) {
        var t = trie
        for (c in w) t = t.getOrPut(c) { Trie() }
        t.w = true
      }
      val cache = mutableMapOf<Int, Int>()
      fun dfs(pos: Int): Int =  if (pos >= s.length) 0 else
        cache.getOrPut(pos) {
          var min = 1 + dfs(pos + 1)
          var t = trie
          for (i in pos..<s.length) {
            t = t[s[i]] ?: break
            if (t.w) min = minOf(min, dfs(i + 1))
          }
          min
        }
      return dfs(0)
    }

```

