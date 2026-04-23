---
layout: leetcode-entry
title: "316. Remove Duplicate Letters"
permalink: "/leetcode/problem/2023-09-26-316-remove-duplicate-letters/"
leetcode_ui: true
entry_slug: "2023-09-26-316-remove-duplicate-letters"
---

[316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/description/) medium
[blog post](https://leetcode.com/problems/remove-duplicate-letters/solutions/4091357/kotlin-greedy-stack/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26092023-316-remove-duplicate-letters?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/9c29040d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/351

#### Problem TLDR

Lexicographical smallest subsequence without duplicates

#### Intuition

The brute force way would be to just consider every position and do a DFS.
To pass the test case, however, there is a greedy way: let's take characters and pop them if new is smaller and the duplicate exists later in a string.

```bash
      // 01234
      //   234
      // bcabc
      // *      b
      //  *     bc
      //   *    a, pop c, pop b
      //    *   ab
      //     *  abc
```

#### Approach

We can use Kotlin's `buildString` API instead of a `Stack`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun removeDuplicateLetters(s: String) = buildString {
      var visited = mutableSetOf<Char>()
      val lastInds = mutableMapOf<Char, Int>()
      s.onEachIndexed { i, c -> lastInds[c] = i}
      s.onEachIndexed { i, c ->
        if (visited.add(c)) {
          while (isNotEmpty() && last() > c && i < lastInds[last()]!!)
            visited.remove(last()).also { setLength(lastIndex) }
          append(c)
        }
      }
    }

```

