---
layout: leetcode-entry
title: "2477. Minimum Fuel Cost to Report to the Capital"
permalink: "/leetcode/problem/2023-02-12-2477-minimum-fuel-cost-to-report-to-the-capital/"
leetcode_ui: true
entry_slug: "2023-02-12-2477-minimum-fuel-cost-to-report-to-the-capital"
---

[2477. Minimum Fuel Cost to Report to the Capital](https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/description/) medium

[blog post](https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/solutions/3175457/kotlin-dfs-with-picture/)

```kotlin
    data class R(val cars: Long, val capacity: Int, val fuel: Long)
    fun minimumFuelCost(roads: Array<IntArray>, seats: Int): Long {
        val nodes = mutableMapOf<Int, MutableList<Int>>()
        roads.forEach { (from, to) ->
            nodes.getOrPut(from, { mutableListOf() }) += to
            nodes.getOrPut(to, { mutableListOf() }) += from
        }
        fun dfs(curr: Int, parent: Int): R {
            val children = nodes[curr]
            if (children == null) return R(1L, seats - 1, 0L)
            var fuel = 0L
            var capacity = 0
            var cars = 0L
            children.filter { it != parent }.forEach {
                val r = dfs(it, curr)
                fuel += r.cars + r.fuel
                capacity += r.capacity
                cars += r.cars
            }
            // seat this passenger
            if (capacity == 0) {
                cars++
                capacity = seats - 1
            } else capacity--
            // optimize cars
            while (capacity - seats >= 0) {
                capacity -= seats
                cars--
            }
            return R(cars, capacity, fuel)
        }
        return dfs(0, 0).fuel
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/116
#### Intuition

![image.png](/assets/leetcode_daily_images/a7237be7.webp)

Let's start from each leaf (node without children). We give `one` car, `seats-1` capacity and `zero` fuel. When children cars arrive, each of them consume `cars` capacity of the fuel. On the hub (node with children), we sat another one passenger, so `capacity--` and we can optimize number of cars arrived, if total `capacity` is more than one car `seats` number.
#### Approach
Use DFS and data class for the result.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(h)$$, h - height of the tree, can be `0..n`

