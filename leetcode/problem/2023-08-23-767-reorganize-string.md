---
layout: leetcode-entry
title: "767. Reorganize String"
permalink: "/leetcode/problem/2023-08-23-767-reorganize-string/"
leetcode_ui: true
entry_slug: "2023-08-23-767-reorganize-string"
---

[767. Reorganize String](https://leetcode.com/problems/reorganize-string/description/) medium
[blog post](https://leetcode.com/problems/reorganize-string/solutions/3948276/kotlin-not-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23082023-767-reorganize-string?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/585ed076.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/317

#### Problem TLDR

Create non repeated subsequent chars string from string

#### Intuition

What will not work:
* naive bubble sort like n^2 algorithm – give false negatives
* n^3 dynamic programming DFS+memo – too slow for the problem

Now, use the hint.

If each time the most frequent char used greedily, solution magically works. (proving that is a homework)

#### Approach

Use Pri0rityQueue to store indices of the frequencies array. If the next char is repeated, and it is the only one left, we have no solution.

#### Complexity

- Time complexity:
$$O(nlog(n))$$, each poll and insert is log(n) in PQ

- Space complexity:
$$O(n)$$, for the result

#### Code

```kotlin

    fun reorganizeString(s: String): String = buildString {
      val freq = IntArray(128)
      s.forEach { freq[it.toInt()]++ }
      val pq = PriorityQueue<Int>(compareBy({ -freq[it] }))
      for (i in 0..127) if (freq[i] > 0) pq.add(i)
      while (pq.isNotEmpty()) {
        var ind = pq.poll()
        if (isNotEmpty() && get(0).toInt() == ind) {
          if (pq.isEmpty()) return ""
          ind = pq.poll().also { pq.add(ind) }
        }
        insert(0, ind.toChar())
        if (--freq[ind] > 0) pq.add(ind)
      }
    }

```

