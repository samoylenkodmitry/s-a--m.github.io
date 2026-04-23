---
layout: leetcode-entry
title: "983. Minimum Cost For Tickets"
permalink: "/leetcode/problem/2023-03-28-983-minimum-cost-for-tickets/"
leetcode_ui: true
entry_slug: "2023-03-28-983-minimum-cost-for-tickets"
---

[983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/description/) medium

[blog post](https://leetcode.com/problems/minimum-cost-for-tickets/solutions/3350465/kotlin-dfs-memo/)

```kotlin

fun mincostTickets(days: IntArray, costs: IntArray): Int {
    val cache = IntArray(days.size) { -1 }
    fun dfs(day: Int): Int {
        if (day >= days.size) return 0
        if (cache[day] != -1) return cache[day]
        var next = day
        while (next < days.size && days[next] - days[day] < 1) next++
        val costOne = costs[0] + dfs(next)
        while (next < days.size && days[next] - days[day] < 7) next++
        val costSeven = costs[1] + dfs(next)
        while (next < days.size && days[next] - days[day] < 30) next++
        val costThirty = costs[2] + dfs(next)
        return minOf(costOne, costSeven, costThirty).also { cache[day] = it}
    }
    return dfs(0)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/162
#### Intuition
For each day we can choose between tickets. Explore all of them and then choose minimum of the cost.

#### Approach
Let's write DFS with memoization algorithm as it is simple to understand.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

