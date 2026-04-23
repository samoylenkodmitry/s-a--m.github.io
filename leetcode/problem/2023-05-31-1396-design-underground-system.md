---
layout: leetcode-entry
title: "1396. Design Underground System"
permalink: "/leetcode/problem/2023-05-31-1396-design-underground-system/"
leetcode_ui: true
entry_slug: "2023-05-31-1396-design-underground-system"
---

[1396. Design Underground System](https://leetcode.com/problems/design-underground-system/description/) medium
[blog post](https://leetcode.com/problems/design-underground-system/solutions/3580723/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/31052023-1396-design-underground?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/229
#### Problem TLDR
Average time `from, to` when different user IDs do `checkIn(from, time1)` and `checkOut(to, time2)`
#### Intuition
Just do what is asked, use `HashMap` to track user's last station.

#### Approach
* store `sum` time and `count` for every `from, to` station
* use `Pair` as key for `HashMap`
#### Complexity
- Time complexity:
$$O(1)$$, for each call
- Space complexity:
$$O(n)$$

#### Code

```kotlin

class UndergroundSystem() {
    val fromToSumTime = mutableMapOf<Pair<String, String>, Long>()
    val fromToCount = mutableMapOf<Pair<String, String>, Int>()
    val idFromTime = mutableMapOf<Int, Pair<String, Int>>()
    fun Pair<String, String>.time() = fromToSumTime[this] ?: 0L
    fun Pair<String, String>.count() = fromToCount[this] ?: 0

    fun checkIn(id: Int, stationName: String, t: Int) {
        idFromTime[id] = stationName to t
    }

    fun checkOut(id: Int, stationName: String, t: Int) {
        val (from, tFrom) = idFromTime[id]!!
        val fromTo = from to stationName
        fromToSumTime[fromTo] = t - tFrom + fromTo.time()
        fromToCount[fromTo] = 1 + fromTo.count()
    }

    fun getAverageTime(startStation: String, endStation: String): Double =
    with(startStation to endStation) {
        time().toDouble() / count().toDouble()
    }

}

```

