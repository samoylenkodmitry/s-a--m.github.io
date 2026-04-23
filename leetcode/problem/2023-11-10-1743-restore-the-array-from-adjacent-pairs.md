---
layout: leetcode-entry
title: "1743. Restore the Array From Adjacent Pairs"
permalink: "/leetcode/problem/2023-11-10-1743-restore-the-array-from-adjacent-pairs/"
leetcode_ui: true
entry_slug: "2023-11-10-1743-restore-the-array-from-adjacent-pairs"
---

[1743. Restore the Array From Adjacent Pairs](https://leetcode.com/problems/restore-the-array-from-adjacent-pairs/description/) medium
[blog post](https://leetcode.com/problems/restore-the-array-from-adjacent-pairs/solutions/4271483/kotlin-graph/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10112023-1743-restore-the-array-from?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/86d2a48f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/399

#### Problem TLDR

Restore an array from adjacent pairs

#### Intuition

We can form an undirected graph and do a Depth-First Search

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun restoreArray(adjacentPairs: Array<IntArray>): IntArray {
      val fromTo = mutableMapOf<Int, MutableList<Int>>()
      for ((from, to) in adjacentPairs) {
        fromTo.getOrPut(from) { mutableListOf() } += to
        fromTo.getOrPut(to) { mutableListOf() } += from
      }
      val visited = HashSet<Int>()
      with(ArrayDeque<Int>()) {
        add(fromTo.keys.first { fromTo[it]!!.size == 1 }!!)
        return IntArray(adjacentPairs.size + 1) {
          while (first() in visited) removeFirst()
          removeFirst().also {
            visited.add(it)
            fromTo[it]?.onEach { add(it) }
          }
        }
      }
    }

```

