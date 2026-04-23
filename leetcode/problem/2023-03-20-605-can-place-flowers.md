---
layout: leetcode-entry
title: "605. Can Place Flowers"
permalink: "/leetcode/problem/2023-03-20-605-can-place-flowers/"
leetcode_ui: true
entry_slug: "2023-03-20-605-can-place-flowers"
---

[605. Can Place Flowers](https://leetcode.com/problems/can-place-flowers/description/) easy

[blog post](https://leetcode.com/problems/can-place-flowers/solutions/3318756/kotlin-greedy/)

```kotlin

fun canPlaceFlowers(flowerbed: IntArray, n: Int): Boolean {
    var count = 0
    if (flowerbed.size == 1 && flowerbed[0] == 0) count++
    if (flowerbed.size >= 2 && flowerbed[0] == 0 && flowerbed[1] == 0) {
        flowerbed[0] = 1
        count++
    }
    for (i in 1..flowerbed.lastIndex - 1) {
        if (flowerbed[i] == 0 && flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0) {
            flowerbed[i] = 1
            count++
        }
    }
    if (flowerbed.size >= 2 && flowerbed[flowerbed.lastIndex] == 0 && flowerbed[flowerbed.lastIndex - 1] == 0) count++
    return count >= n
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/154
#### Intuition
We can plant flowers greedily in every vacant place. This will be the maximum result because if we skip one item, the result is the same for even number of places or worse for odd.

#### Approach
* careful with corner cases
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

