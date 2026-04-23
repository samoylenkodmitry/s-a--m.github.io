---
layout: leetcode-entry
title: "802. Find Eventual Safe States"
permalink: "/leetcode/problem/2023-07-12-802-find-eventual-safe-states/"
leetcode_ui: true
entry_slug: "2023-07-12-802-find-eventual-safe-states"
---

[802. Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/description/) medium
[blog post](https://leetcode.com/problems/find-eventual-safe-states/solutions/3752760/kotlin-dfs/)
[substack](https://dmitriisamoilenko.substack.com/p/13072023-802-find-eventual-safe-states?sd=pf)
![image.png](/assets/leetcode_daily_images/de2e4e8f.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/273
#### Problem TLDR
List of nodes not in cycles
#### Intuition
Simple Depth-First Search will give optimal $$O(n)$$ solution.
When handling the `visited` set, we must separate those in `cycle` and `safe`.
#### Approach
* we can remove from `cycle` set and add to `safe` set in a post-order traversal

#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun eventualSafeNodes(graph: Array<IntArray>): List<Int> {
    val cycle = mutableSetOf<Int>()
        val safe = mutableSetOf<Int>()
            fun cycle(curr: Int): Boolean {
                return if (safe.contains(curr)) false else !cycle.add(curr)
                || graph[curr].any { cycle(it) }
                .also {
                    if (!it) {
                        cycle.remove(curr)
                        safe.add(curr)
                    }
                }
            }
            return graph.indices.filter { !cycle(it) }
        }

```

