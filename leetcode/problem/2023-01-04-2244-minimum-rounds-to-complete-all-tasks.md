---
layout: leetcode-entry
title: "2244. Minimum Rounds to Complete All Tasks"
permalink: "/leetcode/problem/2023-01-04-2244-minimum-rounds-to-complete-all-tasks/"
leetcode_ui: true
entry_slug: "2023-01-04-2244-minimum-rounds-to-complete-all-tasks"
---

[2244. Minimum Rounds to Complete All Tasks](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/description/) medium

[https://t.me/leetcode_daily_unstoppable/74](https://t.me/leetcode_daily_unstoppable/74)

[blog post](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/solutions/2997205/kotlin-dfs-memo/)

```kotlin
    fun minimumRounds(tasks: IntArray): Int {
        val counts = mutableMapOf<Int, Int>()
        tasks.forEach { counts[it] = 1 + counts.getOrDefault(it, 0)}
        var round = 0
        val cache = mutableMapOf<Int, Int>()
        fun fromCount(count: Int): Int {
            if (count == 0) return 0
            if (count < 0 || count == 1) return -1
            return if (count % 3 == 0) {
                count/3
            } else {
                cache.getOrPut(count, {
                    var v = fromCount(count - 3)
                    if (v == -1) v = fromCount(count - 2)
                    if (v == -1) -1 else 1 + v
                })
            }
        }
        counts.values.forEach {
            val rounds = fromCount(it)
            if (rounds == -1) return -1
            round += rounds
        }
        return round
    }

```

For the optimal solution, we must take as many 3's of tasks as possible, then take 2's in any order.
First, we need to count how many tasks of each type there are. Next, we need to calculate the optimal `rounds` for the current tasks type count. There is a math solution, but ultimately we just can do DFS

Space: O(N), Time: O(N), counts range is always less than N

