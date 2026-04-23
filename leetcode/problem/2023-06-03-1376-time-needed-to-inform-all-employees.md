---
layout: leetcode-entry
title: "1376. Time Needed to Inform All Employees"
permalink: "/leetcode/problem/2023-06-03-1376-time-needed-to-inform-all-employees/"
leetcode_ui: true
entry_slug: "2023-06-03-1376-time-needed-to-inform-all-employees"
---

[1376. Time Needed to Inform All Employees](https://leetcode.com/problems/time-needed-to-inform-all-employees/description/) medium
[blog post](https://leetcode.com/problems/time-needed-to-inform-all-employees/solutions/3591362/kotlin-dfs/)
[substack](https://dmitriisamoilenko.substack.com/p/03062023-1376-time-needed-to-inform?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/234
#### Problem TLDR
Total `time` from `headID` to all nodes in graph.
#### Intuition
Total time will be the maximum time from the root of the graph to the lowest node. To find it out, we can use DFS.
#### Approach
Build the graph, then write the DFS.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun numOfMinutes(n: Int, headID: Int, manager: IntArray, informTime: IntArray): Int {
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        (0 until n).forEach { fromTo.getOrPut(manager[it]) { mutableListOf() } += it }
        fun dfs(curr: Int): Int {
            return informTime[curr] + (fromTo[curr]?.map { dfs(it) }?.max() ?: 0)
        }
        return dfs(headID)
    }

```

