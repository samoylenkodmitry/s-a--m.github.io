---
layout: leetcode-entry
title: "207. Course Schedule"
permalink: "/leetcode/problem/2023-07-13-207-course-schedule/"
leetcode_ui: true
entry_slug: "2023-07-13-207-course-schedule"
---

[207. Course Schedule](https://leetcode.com/problems/course-schedule/description/) medium
[blog post](https://leetcode.com/problems/course-schedule/solutions/3757355/kotlin-toposort-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/13072023-207-course-schedule?sd=pf)
![image.png](/assets/leetcode_daily_images/ab496b3d.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/274
#### Problem TLDR
If `none` edges in a cycle
#### Intuition
To detect cycle, we can use DFS and two sets `cycle` and `safe`. Or use Topological Sort and check that all elements are visited.

#### Approach
Let's use Topological Sort with Breadth-First Search.
* build `indegree` - number of input nodes for each node
* add to BFS only nodes with `indegree[node] == 0`
* decrease `indegree` as it visited

#### Complexity
- Time complexity:
$$O(VE)$$

- Space complexity:
$$O(E + V)$$

#### Code

```kotlin

fun canFinish(numCourses: Int, prerequisites: Array<IntArray>): Boolean {
    val fromTo = mutableMapOf<Int, MutableSet<Int>>()
        val indegree = IntArray(numCourses)
        prerequisites.forEach { (to, from) ->
            fromTo.getOrPut(from) { mutableSetOf() } += to
            indegree[to]++
        }
        return with(ArrayDeque<Int>()) {
            addAll((0 until numCourses).filter { indegree[it] == 0 })
            generateSequence { if (isEmpty()) null else poll() }.map {
                fromTo[it]?.forEach {
                    if (--indegree[it] == 0) add(it)
                }
            }.count() == numCourses
        }
    }

```

