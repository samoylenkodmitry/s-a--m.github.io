---
layout: leetcode-entry
title: "875. Koko Eating Bananas"
permalink: "/leetcode/problem/2023-03-08-875-koko-eating-bananas/"
leetcode_ui: true
entry_slug: "2023-03-08-875-koko-eating-bananas"
---

[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/description/) medium

[blog post](https://leetcode.com/problems/koko-eating-bananas/solutions/3271497/kotlin-binary-search/)

```kotlin

fun minEatingSpeed(piles: IntArray, h: Int): Int {
    fun canEatAll(speed: Long): Boolean {
        var time = 0L
        piles.forEach {
            time += (it.toLong() / speed) + if ((it.toLong() % speed) == 0L) 0L else 1L
        }
        return time <= h
    }
    var lo = 1L
    var hi = piles.asSequence().map { it.toLong() }.sum()!!
    var minSpeed = hi
    while (lo <= hi) {
        val speed = lo + (hi - lo) / 2
        if (canEatAll(speed)) {
            minSpeed = minOf(minSpeed, speed)
            hi = speed - 1
        } else {
            lo = speed + 1
        }
    }
    return minSpeed.toInt()
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/142
#### Intuition
Given the `speed` we can count how many `hours` take Coco to eat all the bananas. With growth of `speed` `hours` growth too, so we can binary search in that space.

#### Approach
For more robust binary search:
* use inclusive condition check `lo == hi`
* always move boundaries `mid + 1`, `mid - 1`
* compute the result on each step
#### Complexity
- Time complexity:
$$O(nlog_2(m))$$, `m` - is `hours` range
- Space complexity:
$$O(1)$$

