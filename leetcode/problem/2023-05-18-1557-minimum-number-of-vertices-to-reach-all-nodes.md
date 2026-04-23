---
layout: leetcode-entry
title: "1557. Minimum Number of Vertices to Reach All Nodes"
permalink: "/leetcode/problem/2023-05-18-1557-minimum-number-of-vertices-to-reach-all-nodes/"
leetcode_ui: true
entry_slug: "2023-05-18-1557-minimum-number-of-vertices-to-reach-all-nodes"
---

[1557. Minimum Number of Vertices to Reach All Nodes](https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/solutions/3536694/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/18052023-1557-minimum-number-of-vertices?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/216
#### Problem TLDR
Find all starting nodes in graph.
#### Intuition
Count nodes that have no incoming connections.

#### Approach
* we can use subtract operation in Kotlin
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun findSmallestSetOfVertices(n: Int, edges: List<List<Int>>): List<Int> =
    (0 until n) - edges.map { it[1] }

```

