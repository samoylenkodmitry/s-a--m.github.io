---
layout: leetcode-entry
title: "97. Interleaving String"
permalink: "/leetcode/problem/2023-08-25-97-interleaving-string/"
leetcode_ui: true
entry_slug: "2023-08-25-97-interleaving-string"
---

[97. Interleaving String](https://leetcode.com/problems/interleaving-string/description/) medium
[blog post](https://leetcode.com/problems/interleaving-string/solutions/3956738/kotlin-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25082023-97-interleaving-string?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/48810b28.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/319

#### Problem TLDR

Can a string be a merge of two other strings

#### Intuition

Do DFS with two pointers, each time taking a char from the first or the second's string, the third pointer will be `p1 + p2`. The result will depend only on the remaining suffixes, so can be safely cached.

#### Approach

* calculate the key into a single Int `p1 + p2 * 100`
* check that lengths are adding up

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun isInterleave(s1: String, s2: String, s3: String): Boolean {
      val cache = mutableMapOf<Int, Boolean>()
      fun dfs(p1: Int, p2: Int): Boolean = cache.getOrPut(p1 + p2 * 100) {
        p1 < s1.length && p2 < s2.length && (
          s1[p1] == s3[p1 + p2] && dfs(p1 + 1, p2)
          || s2[p2] == s3[p1 + p2] && dfs(p1, p2 + 1)
        )
        || p1 == s1.length && s2.substring(p2) == s3.substring(p1 + p2)
        || p2 == s2.length && s1.substring(p1) == s3.substring(p1 + p2)
      }
      return s1.length + s2.length == s3.length && dfs(0, 0)
    }

```

