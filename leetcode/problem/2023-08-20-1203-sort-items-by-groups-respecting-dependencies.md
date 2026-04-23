---
layout: leetcode-entry
title: "1203. Sort Items by Groups Respecting Dependencies"
permalink: "/leetcode/problem/2023-08-20-1203-sort-items-by-groups-respecting-dependencies/"
leetcode_ui: true
entry_slug: "2023-08-20-1203-sort-items-by-groups-respecting-dependencies"
---

[1203. Sort Items by Groups Respecting Dependencies](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/description/) hard
[blog post](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/solutions/3935139/kotlin-idea-tricks/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20082023-1203-sort-items-by-groups?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/95f9efbc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/314

#### Problem TLDR

Sort items by groups and in groups given dependencies.

#### Intuition

Use hint.

We can split items by groups and check groups dependencies.
Next, do Topological Sort for groups and then do Topological Sort for items in each group.

#### Approach

Now, the tricks:
* if we consider each `-1` as a separate group, code will become cleaner
* we don't have to do separate Topological Sort for each group, just sort whole graph of items, then filter by each group
* cycle detection can be done in a Topological Sort: if there is a cycle, there is no item with `indegree == 0`
* Topological Sort function can be reused

#### Complexity

- Time complexity:
$$O(nm + E)$$

- Space complexity:
$$O(n + n + E)$$

#### Code

```kotlin

    class G(count: Int, val fromTo: MutableMap<Int, MutableSet<Int>> = mutableMapOf()) {
      operator fun get(k: Int) = fromTo.getOrPut(k) { mutableSetOf() }
      val order: List<Int> by lazy {
        val indegree = IntArray(count)
        fromTo.values.onEach { it.onEach { indegree[it]++ } }
        val queue = ArrayDeque<Int>(indegree.indices.filter { indegree[it] == 0 })
        generateSequence { queue.poll() }
            .onEach { fromTo[it]?.onEach { if (--indegree[it] == 0) queue += it } }
            .toList().takeIf { it.size == count } ?: listOf()
      }
    }
    fun sortItems(n: Int, m: Int, group: IntArray, beforeItems: List<List<Int>>): IntArray {
      var groupsCount = m
      for (i in 0 until n) if (group[i] == -1) group[i] = groupsCount++
      val items = G(n)
      val groups = G(groupsCount)
      for (to in beforeItems.indices)
        for (from in beforeItems[to])
          if (group[to] == group[from]) items[from] += to
          else groups[group[from]] += group[to]
      return groups.order.flatMap { g -> items.order.filter { group[it] == g } }.toIntArray()
    }

```

