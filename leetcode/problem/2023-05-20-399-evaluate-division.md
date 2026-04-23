---
layout: leetcode-entry
title: "399. Evaluate Division"
permalink: "/leetcode/problem/2023-05-20-399-evaluate-division/"
leetcode_ui: true
entry_slug: "2023-05-20-399-evaluate-division"
---

[399. Evaluate Division](https://leetcode.com/problems/evaluate-division/description/) medium
[blog post](https://leetcode.com/problems/evaluate-division/solutions/3543427/kotlin-n-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/20052023-399-evaluate-division?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/218
#### Problem TLDR
Given values for `a/b` and `b/c` find answers for `a/c`.
#### Intuition
Let's build a graph, `a` -> `b` with weights of `values[a/b]`. Then answer is a path from one node to the other. The shortest path can be found with a Breadth-First Search.

#### Approach
* careful with corner case `x/x`, where `x` is not in a graph.
#### Complexity
- Time complexity:
$$O(nEV)$$
- Space complexity:
$$O(n+E+V)$$

#### Code

```kotlin

fun calcEquation(equations: List<List<String>>, values: DoubleArray, queries: List<List<String>>): DoubleArray {
    val fromTo = mutableMapOf<String, MutableList<Pair<String, Double>>>()
    equations.forEachIndexed { i, (from, to) ->
        fromTo.getOrPut(from) { mutableListOf() } += to to values[i]
        fromTo.getOrPut(to) { mutableListOf() } += from to (1.0 / values[i])
    }
    // a/c = a/b * b/c
    return queries.map { (from, to) ->
        with(ArrayDeque<Pair<String, Double>>()) {
            val visited = HashSet<String>()
                visited.add(from)
                if (fromTo.containsKey(to)) add(from to 1.0)
                while (isNotEmpty()) {
                    repeat(size) {
                        val (point, value) = poll()
                        if (point == to) return@map value
                        fromTo[point]?.forEach { (next, nvalue) ->
                            if (visited.add(next)) add(next to value * nvalue)
                        }
                    }
                }
                -1.0
            }
        }.toDoubleArray()
    }

```

