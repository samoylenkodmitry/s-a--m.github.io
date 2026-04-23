---
layout: leetcode-entry
title: "2050. Parallel Courses III"
permalink: "/leetcode/problem/2023-10-18-2050-parallel-courses-iii/"
leetcode_ui: true
entry_slug: "2023-10-18-2050-parallel-courses-iii"
---

[2050. Parallel Courses III](https://leetcode.com/problems/parallel-courses-iii/description/) hard
[blog post](https://leetcode.com/problems/parallel-courses-iii/solutions/4180807/kotlin-dfs-memo-from-leafs/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18102023-2050-parallel-courses-iii?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/2c345d6d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/374

#### Problem TLDR

Shortest `time` to visit all nodes in `relations=[from, to]` graph

#### Intuition

We can start from nodes without `out` siblings - leafs and do Depth-First Search from them, calculating time for each sibling in parallel and choosing the maximum. That is an optimal way to visit all the nodes. For each node, a solution can be cached.

#### Approach

Let's use some [Kotlin's API](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/):
* calculate leafs by subtracting all `from` nodes from all the nodes `1..n`
* form a graph `Map<Int, List<Int>>` by using `groupBy`
* choose the maximum and return it with `maxOf`
* get and put to map with `getOrPut`

#### Complexity

- Time complexity:
$$O(nr)$$, will visit each node only once, r - average siblings count for each node

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minimumTime(n: Int, relations: Array<IntArray>, time: IntArray): Int {
      val lastNodes = (1..n) - relations.map { it[0] }
      val fromTo = relations.groupBy({ it[1] }, { it[0] })
      val cache = mutableMapOf<Int, Int>()
      fun dfs(curr: Int): Int = cache.getOrPut(curr) {
        time[curr - 1] + (fromTo[curr]?.maxOf { dfs(it) } ?: 0)
      }
      return lastNodes.maxOf { dfs(it) }
    }

```

P.S.: we can also just choose the maximum, as it will be the longest path:

```kotlin
    fun minimumTime(n: Int, relations: Array<IntArray>, time: IntArray): Int {
      val fromTo = relations.groupBy({ it[1] }, { it[0] })
      val cache = mutableMapOf<Int, Int>()
      fun dfs(curr: Int): Int = cache.getOrPut(curr) {
        time[curr - 1] + (fromTo[curr]?.maxOf { dfs(it) } ?: 0)
      }
      return (1..n).maxOf { dfs(it) }
    }
```

